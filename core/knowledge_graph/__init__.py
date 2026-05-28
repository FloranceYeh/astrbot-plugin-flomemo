import asyncio
import json
import re
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context

DEFAULT_GRAPH_PROMPT = (
    "请从以下 TL;DR 中抽取人物关系、事件因果与关键事实，输出 JSON 数组。"
    "每个元素包含：source, relation, target, type, evidence。\n"
    "仅输出 JSON，不要额外说明。\n\n"
    "TL;DR：\n{summary}"
)


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower())
    return {token for token in tokens if token}


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
        self._nodes: set[str] = set()
        self._edges: list[dict[str, Any]] = []

    async def load(self):
        async with self._lock:
            graph = await self._load_json_file(self._path, {"nodes": [], "edges": []})
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            self._nodes = set(nodes) if isinstance(nodes, list) else set()
            self._edges = edges if isinstance(edges, list) else []

    async def save(self):
        async with self._lock:
            data = {"nodes": sorted(self._nodes), "edges": list(self._edges)}
        await self._save_json_file(self._path, data)

    async def update_from_summary(self, session_id: str, date_str: str, summary: str):
        provider_id = await self._resolve_chat_provider_id(session_id)
        if not provider_id:
            logger.warning(f"无法获取 LLM provider，跳过 {session_id} 的图谱抽取。")
            return
        prompt_template = self.config.get("graph_prompt", DEFAULT_GRAPH_PROMPT)
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
                source = str(item.get("source", "")).strip()
                target = str(item.get("target", "")).strip()
                relation = str(item.get("relation", "")).strip()
                if not source or not target or not relation:
                    continue
                self._nodes.add(source)
                self._nodes.add(target)
                self._edges.append(
                    {
                        "source": source,
                        "relation": relation,
                        "target": target,
                        "type": str(item.get("type", "")).strip(),
                        "evidence": str(item.get("evidence", "")).strip(),
                        "date": date_str,
                        "session_id": session_id,
                    }
                )
        await self.save()

    async def query(self, query: str) -> list[dict[str, Any]]:
        keywords = _tokenize(query)
        async with self._lock:
            edges = list(self._edges)
        if not keywords:
            return edges[:10]
        filtered = [
            edge
            for edge in edges
            if any(
                keyword in str(edge.get("source", "")).lower()
                or keyword in str(edge.get("target", "")).lower()
                for keyword in keywords
            )
        ]
        return filtered

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
