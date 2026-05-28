import asyncio
import inspect
import json
import math
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context


def _now_ts() -> float:
    return time.time()


def _today_str(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.date().isoformat()


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower())
    return {token for token in tokens if token}


def _lexical_similarity(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a_list = list(vec_a)
    b_list = list(vec_b)
    if not a_list or not b_list or len(a_list) != len(b_list):
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(y * y for y in b_list))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class WorkingMemoryStore:
    def __init__(self, context: Context, config: AstrBotConfig, data_dir: Path):
        self.context = context
        self.config = config
        self._path = data_dir / "working_memory.json"
        self._lock = asyncio.Lock()
        self._entries: list[dict[str, Any]] = []
        self._embedding_provider: Any = None

    async def load(self):
        async with self._lock:
            self._entries = await self._load_json_file(self._path, [])
            self._prune()

    async def save(self):
        async with self._lock:
            data = list(self._entries)
        await self._save_json_file(self._path, data)

    async def add_message(self, session_id: str, role: str, content: str):
        content = content.strip()
        if not content:
            return
        embedding = await self._embed_text(content)
        entry = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "embedding": embedding,
            "ts": _now_ts(),
            "date": _today_str(),
        }
        async with self._lock:
            self._entries.append(entry)
            self._prune()
        await self.save()

    async def query(self, session_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        query_embedding = await self._embed_text(query)
        async with self._lock:
            entries = [item for item in self._entries if item.get("session_id") == session_id]
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in entries:
            content = str(item.get("content", ""))
            embedding = item.get("embedding")
            if query_embedding and isinstance(embedding, list):
                score = _cosine_similarity(query_embedding, embedding)
            else:
                score = _lexical_similarity(query, content)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    async def get_entries_for_date(self, date_str: str) -> list[dict[str, Any]]:
        async with self._lock:
            return [item for item in self._entries if item.get("date") == date_str]

    async def reset_session(self, session_id: str):
        async with self._lock:
            self._entries = [
                item for item in self._entries if item.get("session_id") != session_id
            ]
        await self.save()

    async def count(self) -> int:
        async with self._lock:
            return len(self._entries)

    def _prune(self):
        retention_days = self._get_config_int("retention_days", 3, minimum=1)
        cutoff = _now_ts() - retention_days * 86400
        self._entries = [item for item in self._entries if item.get("ts", 0.0) >= cutoff]

    async def _embed_text(self, text: str) -> list[float] | None:
        provider = self._resolve_embedding_provider()
        if not provider:
            logger.warning("未找到 Embedding Provider，记忆将使用关键词相似度回退。")
            return None
        if hasattr(provider, "embed_texts"):
            method = provider.embed_texts
            result = method([text])
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, list) and result:
                return result[0]
        if hasattr(provider, "get_embedding"):
            method = provider.get_embedding
            result = method(text)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, list):
                return result
        logger.warning("Embedding Provider 不支持 embed_texts 或 get_embedding。")
        return None

    def _resolve_embedding_provider(self) -> Any:
        if self._embedding_provider:
            return self._embedding_provider
        provider_id = str(self.config.get("embedding_provider_id", "")).strip()
        provider = None
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
        if not provider:
            providers = []
            try:
                providers = self.context.get_all_embedding_providers()
            except AttributeError:
                providers = []
            if providers:
                provider = providers[0]
        if provider:
            self._embedding_provider = provider
            return provider
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

    def _get_config_int(self, key: str, default: int, minimum: int | None = None) -> int:
        value = self.config.get(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
        if minimum is not None and value < minimum:
            return minimum
        return value
