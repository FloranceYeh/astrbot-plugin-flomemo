import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context

DEFAULT_GRAPH_PROMPT = (
    "请从以下 TL;DR 中抽取人物关系、事件因果与关键事实，输出 JSON 数组。"
    "每个元素包含：source, relation, target, type, evidence, confidence。\n"
    "confidence 为 0 到 1 的浮点数，表示该关系的可信度。\n"
    "仅输出 JSON，不要额外说明。\n\n"
    "TL;DR：\n{summary}"
)
DEFAULT_GRAPH_MIN_CONFIDENCE = 0.35
DEFAULT_GRAPH_QUERY_LIMIT = 10


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower())
    return {token for token in tokens if token}


def _now_ts() -> float:
    return time.time()


def _normalize_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value.strip())
    return text


def _normalize_entity_key(value: str) -> str:
    text = _normalize_text(value).casefold()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def _normalize_relation_key(value: str) -> str:
    text = _normalize_text(value).casefold()
    text = re.sub(r"\s+", "_", text)
    return text


def _dedupe_text_chunks(*values: str) -> str:
    chunks: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_text(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        chunks.append(normalized)
    return " | ".join(chunks)


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning(f"知识图谱 JSON 解析失败: {exc}")
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


class KnowledgeGraphStore:
    def __init__(self, context: Context, config: AstrBotConfig, data_dir: Path):
        self.context = context
        self.config = config
        self._path = data_dir / "knowledge_graph.json"
        self._lock = asyncio.Lock()
        self._nodes: dict[str, str] = {}
        self._edges: list[dict[str, Any]] = []

    async def load(self):
        async with self._lock:
            graph = await self._load_json_file(self._path, {"nodes": [], "edges": []})
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            self._nodes = self._load_nodes(nodes)
            raw_edges = edges if isinstance(edges, list) else []
            self._edges = self._normalize_loaded_edges(raw_edges)

    async def save(self):
        async with self._lock:
            data = {
                "nodes": [
                    {"key": key, "name": name} for key, name in sorted(self._nodes.items())
                ],
                "edges": list(self._edges),
            }
        await self._save_json_file(self._path, data)

    async def update_from_summary(self, session_id: str, date_str: str, summary: str):
        provider_id = await self._resolve_chat_provider_id(session_id)
        if not provider_id:
            logger.warning(f"无法获取 LLM provider，跳过 {session_id} 的图谱抽取。")
            return
        prompt_template = self._get_graph_config("graph_prompt", DEFAULT_GRAPH_PROMPT)
        prompt = str(prompt_template).format(summary=summary)
        llm_resp = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=prompt,
        )
        payload = _extract_json_array(getattr(llm_resp, "completion_text", ""))
        if not payload:
            return
        async with self._lock:
            for item in payload:
                edge = self._build_edge(item, session_id, date_str)
                if not edge:
                    continue
                self._merge_edge(edge)
        await self.save()

    async def query(self, query: str) -> list[dict[str, Any]]:
        keywords = _tokenize(query)
        async with self._lock:
            edges = list(self._edges)
        min_confidence = self._get_graph_float(
            "graph_min_confidence", DEFAULT_GRAPH_MIN_CONFIDENCE, minimum=0.0, maximum=1.0
        )
        filtered = [
            edge
            for edge in edges
            if float(edge.get("confidence", 0.0)) >= min_confidence
        ]
        if keywords:
            filtered = [
                edge
                for edge in filtered
                if self._edge_match_keywords(edge, keywords)
            ]

        ranked = sorted(
            filtered,
            key=lambda edge: (
                self._edge_relevance_score(query, keywords, edge),
                float(edge.get("updated_at", edge.get("created_at", 0.0))),
            ),
            reverse=True,
        )
        limit = self._get_graph_int("graph_query_limit", DEFAULT_GRAPH_QUERY_LIMIT, minimum=1)
        return ranked[:limit]

    async def count_nodes(self) -> int:
        async with self._lock:
            return len(self._nodes)

    async def count_edges(self) -> int:
        async with self._lock:
            return len(self._edges)

    async def _resolve_chat_provider_id(self, session_id: str) -> str | None:
        provider_id = str(self.config.get("llm_provider_id", "")).strip()
        if provider_id:
            return provider_id
        try:
            return await self.context.get_current_chat_provider_id(umo=session_id)
        except AttributeError:
            logger.warning("当前 AstrBot 版本不支持 get_current_chat_provider_id。")
            return None

    async def _load_json_file(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except OSError as exc:
            logger.error(f"读取数据文件失败: {path} ({exc})")
            return default
        if not content.strip():
            return default
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error(f"解析数据文件失败: {path} ({exc})")
            return default

    async def _save_json_file(self, path: Path, data: Any):
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            await asyncio.to_thread(tmp_path.write_text, payload, encoding="utf-8")
            await asyncio.to_thread(tmp_path.replace, path)
        except OSError as exc:
            logger.error(f"写入数据文件失败: {path} ({exc})")

    def _get_graph_config(self, key: str, default: Any) -> Any:
        container = self.config.get("graph", {})
        if isinstance(container, dict):
            return container.get(key, default)
        return default

    def _get_graph_int(self, key: str, default: int, minimum: int | None = None) -> int:
        value = self._get_graph_config(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
        if minimum is not None and value < minimum:
            return minimum
        return value

    def _get_graph_float(
        self,
        key: str,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        value = self._get_graph_config(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
        if minimum is not None and value < minimum:
            value = minimum
        if maximum is not None and value > maximum:
            value = maximum
        return value

    def _load_nodes(self, nodes: Any) -> dict[str, str]:
        if not isinstance(nodes, list):
            return {}
        loaded: dict[str, str] = {}
        for item in nodes:
            if isinstance(item, dict):
                name = _normalize_text(str(item.get("name", "")))
                key = _normalize_entity_key(str(item.get("key", "")) or name)
            else:
                name = _normalize_text(str(item))
                key = _normalize_entity_key(name)
            if not key or not name:
                continue
            loaded[key] = self._select_display_name(loaded.get(key), name)
        return loaded

    def _normalize_loaded_edges(self, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in edges:
            edge = self._build_edge(
                item,
                str(item.get("session_id", "")),
                str(item.get("date", "")),
                preserve_timestamps=True,
            )
            if not edge:
                continue
            edge["created_at"] = self._safe_float(item.get("created_at"), edge["created_at"])
            edge["updated_at"] = self._safe_float(item.get("updated_at"), edge["updated_at"])
            key = str(edge["key"])
            current = merged.get(key)
            if current is None:
                merged[key] = edge
            else:
                merged[key] = self._combine_edges(current, edge)
        self._rebuild_nodes_from_edges(list(merged.values()))
        return list(merged.values())

    def _build_edge(
        self,
        item: dict[str, Any],
        session_id: str,
        date_str: str,
        preserve_timestamps: bool = False,
    ) -> dict[str, Any] | None:
        raw_source = _normalize_text(str(item.get("source", "")))
        raw_target = _normalize_text(str(item.get("target", "")))
        raw_relation = _normalize_text(str(item.get("relation", "")))
        if not raw_source or not raw_target or not raw_relation or not session_id or not date_str:
            return None

        source_key = _normalize_entity_key(raw_source)
        target_key = _normalize_entity_key(raw_target)
        relation_key = _normalize_relation_key(raw_relation)
        if not source_key or not target_key or not relation_key:
            return None

        source_name = self._register_node(source_key, raw_source)
        target_name = self._register_node(target_key, raw_target)
        confidence = self._safe_confidence(item.get("confidence"))
        now_ts = self._safe_float(item.get("updated_at"), _now_ts()) if preserve_timestamps else _now_ts()
        created_at = (
            self._safe_float(item.get("created_at"), now_ts) if preserve_timestamps else now_ts
        )
        return {
            "key": self._make_edge_key(source_key, relation_key, target_key, date_str, session_id),
            "source": source_name,
            "source_key": source_key,
            "relation": raw_relation,
            "relation_key": relation_key,
            "target": target_name,
            "target_key": target_key,
            "type": _normalize_text(str(item.get("type", ""))),
            "evidence": _normalize_text(str(item.get("evidence", ""))),
            "confidence": confidence,
            "date": date_str,
            "session_id": session_id,
            "created_at": created_at,
            "updated_at": now_ts,
        }

    def _merge_edge(self, edge: dict[str, Any]):
        key = str(edge["key"])
        for index, current in enumerate(self._edges):
            if str(current.get("key", "")) != key:
                continue
            self._edges[index] = self._combine_edges(current, edge)
            return
        self._edges.append(edge)

    def _combine_edges(self, current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        source_name = self._register_node(str(current.get("source_key", "")), str(incoming.get("source", current.get("source", ""))))
        target_name = self._register_node(str(current.get("target_key", "")), str(incoming.get("target", current.get("target", ""))))
        confidence = max(
            self._safe_confidence(current.get("confidence")),
            self._safe_confidence(incoming.get("confidence")),
        )
        return {
            **current,
            "source": source_name,
            "target": target_name,
            "type": _dedupe_text_chunks(str(current.get("type", "")), str(incoming.get("type", ""))),
            "evidence": _dedupe_text_chunks(
                str(current.get("evidence", "")),
                str(incoming.get("evidence", "")),
            ),
            "confidence": confidence,
            "updated_at": max(
                self._safe_float(current.get("updated_at"), 0.0),
                self._safe_float(incoming.get("updated_at"), 0.0),
            ),
        }

    def _register_node(self, key: str, name: str) -> str:
        display = self._select_display_name(self._nodes.get(key), name)
        self._nodes[key] = display
        return display

    def _select_display_name(self, current: str | None, candidate: str) -> str:
        normalized_candidate = _normalize_text(candidate)
        if not normalized_candidate:
            return _normalize_text(current or "")
        normalized_current = _normalize_text(current or "")
        if not normalized_current:
            return normalized_candidate
        if len(normalized_candidate) > len(normalized_current):
            return normalized_candidate
        return normalized_current

    def _rebuild_nodes_from_edges(self, edges: list[dict[str, Any]]):
        rebuilt: dict[str, str] = {}
        for edge in edges:
            source_key = str(edge.get("source_key", ""))
            target_key = str(edge.get("target_key", ""))
            source_name = _normalize_text(str(edge.get("source", "")))
            target_name = _normalize_text(str(edge.get("target", "")))
            if source_key and source_name:
                rebuilt[source_key] = self._select_display_name(rebuilt.get(source_key), source_name)
            if target_key and target_name:
                rebuilt[target_key] = self._select_display_name(rebuilt.get(target_key), target_name)
        self._nodes = rebuilt

    def _make_edge_key(
        self,
        source_key: str,
        relation_key: str,
        target_key: str,
        date_str: str,
        session_id: str,
    ) -> str:
        return "|".join([source_key, relation_key, target_key, date_str, session_id])

    def _safe_confidence(self, value: Any) -> float:
        return self._safe_float(value, 0.7, minimum=0.0, maximum=1.0)

    def _safe_float(
        self,
        value: Any,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = default
        if minimum is not None and result < minimum:
            result = minimum
        if maximum is not None and result > maximum:
            result = maximum
        return result

    def _edge_match_keywords(self, edge: dict[str, Any], keywords: set[str]) -> bool:
        haystacks = [
            str(edge.get("source", "")),
            str(edge.get("target", "")),
            str(edge.get("relation", "")),
            str(edge.get("type", "")),
            str(edge.get("evidence", "")),
        ]
        lowered = [item.casefold() for item in haystacks if item]
        return any(keyword in item for keyword in keywords for item in lowered)

    def _edge_relevance_score(
        self, query: str, keywords: set[str], edge: dict[str, Any]
    ) -> float:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        relation = str(edge.get("relation", ""))
        edge_text = " ".join(
            [
                source,
                relation,
                target,
                str(edge.get("type", "")),
                str(edge.get("evidence", "")),
            ]
        )
        lexical_score = len(keywords & _tokenize(edge_text)) if keywords else 0
        source_target_score = _tokenize(" ".join([source, target]))
        entity_overlap = len(keywords & source_target_score) if keywords else 0
        similarity = 0.0
        query_tokens = _tokenize(query)
        edge_tokens = _tokenize(edge_text)
        if query_tokens and edge_tokens:
            similarity = len(query_tokens & edge_tokens) / len(query_tokens | edge_tokens)
        confidence = self._safe_confidence(edge.get("confidence"))
        recency = self._edge_recency_bonus(edge)
        return lexical_score * 2.0 + entity_overlap * 3.0 + similarity + confidence + recency

    def _edge_recency_bonus(self, edge: dict[str, Any]) -> float:
        updated_at = self._safe_float(edge.get("updated_at"), 0.0)
        if updated_at <= 0:
            return 0.0
        age_seconds = max(_now_ts() - updated_at, 0.0)
        horizon_seconds = 30 * 86400
        freshness = 1.0 - min(age_seconds / horizon_seconds, 1.0)
        return freshness * 0.5
