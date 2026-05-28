import asyncio
import inspect
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus.exceptions import MilvusException
from pymilvus.milvus_client.index import IndexParams


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
    norm_a = sum(x * x for x in a_list) ** 0.5
    norm_b = sum(y * y for y in b_list) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _escape_expr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class WorkingMemoryStore:
    def __init__(self, context: Context, config: AstrBotConfig, data_dir: Path):
        self.context = context
        self.config = config
        self._data_dir = data_dir
        self._lock = asyncio.Lock()
        self._embedding_provider: Any = None
        self._client: MilvusClient | None = None
        self._collection_name: str | None = None
        self._connect_failed = False

    async def load(self):
        await self._ensure_collection()
        await self._prune()

    async def save(self):
        await self._ensure_collection()

    async def add_message(self, session_id: str, role: str, content: str) -> bool:
        content = content.strip()
        if not content:
            return False
        embedding = await self._embed_text(content)
        if embedding is None:
            logger.warning("未生成 embedding，跳过工作记忆写入。")
            return False
        await self._ensure_collection(dim=len(embedding))
        if not self._client or not self._collection_name:
            return False
        payload = [
            {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "role": role,
                "content": content,
                "date": _today_str(),
                "ts": _now_ts(),
                "embedding": embedding,
            }
        ]
        async with self._lock:
            await asyncio.to_thread(
                self._client.insert,
                self._collection_name,
                payload,
            )
        await self._prune()
        return True

    async def query(self, session_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        await self._ensure_collection()
        if not self._client or not self._collection_name:
            return []
        query_embedding = await self._embed_text(query)
        expr = f'session_id == "{_escape_expr_value(session_id)}"'
        if query_embedding is not None:
            search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
            async with self._lock:
                results = await asyncio.to_thread(
                    self._client.search,
                    self._collection_name,
                    data=[query_embedding],
                    filter=expr,
                    limit=top_k,
                    output_fields=["session_id", "role", "content", "date", "ts"],
                    search_params=search_params,
                    anns_field="embedding",
                )
            return self._format_search_results(results)

        async with self._lock:
            rows = await asyncio.to_thread(
                self._client.query,
                self._collection_name,
                filter=expr,
                output_fields=["session_id", "role", "content", "date", "ts", "embedding"],
            )
        return self._fallback_rank(query, rows, top_k)

    async def get_entries_for_date(self, date_str: str) -> list[dict[str, Any]]:
        await self._ensure_collection()
        if not self._client or not self._collection_name:
            return []
        expr = f'date == "{_escape_expr_value(date_str)}"'
        async with self._lock:
            rows = await asyncio.to_thread(
                self._client.query,
                self._collection_name,
                filter=expr,
                output_fields=["session_id", "role", "content", "date", "ts", "embedding"],
            )
        return rows

    async def reset_session(self, session_id: str):
        await self._ensure_collection()
        if not self._client or not self._collection_name:
            return
        expr = f'session_id == "{_escape_expr_value(session_id)}"'
        async with self._lock:
            await asyncio.to_thread(
                self._client.delete,
                self._collection_name,
                filter=expr,
            )

    async def count(self) -> int:
        await self._ensure_collection()
        if not self._client or not self._collection_name:
            return 0
        stats = await asyncio.to_thread(
            self._client.get_collection_stats, self._collection_name
        )
        if not isinstance(stats, dict):
            return 0
        count_value = stats.get("row_count", 0)
        try:
            return int(count_value)
        except (TypeError, ValueError):
            return 0

    async def _ensure_client(self):
        if self._client is not None:
            return True
        if self._connect_failed:
            return False
        try:
            await asyncio.to_thread(self._connect)
        except (MilvusException, OSError, ValueError) as exc:
            logger.error(f"Milvus 连接失败: {exc}")
            self._client = None
            self._connect_failed = True
            return False
        return self._client is not None

    def _connect(self):
        lite_path = str(self._get_milvus_config("lite_path", "")).strip()
        address = str(self._get_milvus_config("address", "")).strip()
        secure = self._get_milvus_bool("secure", False)
        if not lite_path and not address:
            lite_path = "milvus/flomemo.db"
        if lite_path:
            resolved = Path(lite_path)
            if not resolved.is_absolute():
                resolved = self._data_dir / resolved
            resolved.parent.mkdir(parents=True, exist_ok=True)
            uri = str(resolved)
        elif address:
            if address.startswith("http://") or address.startswith("https://"):
                uri = address
            else:
                scheme = "https" if secure else "http"
                uri = f"{scheme}://{address}"
        else:
            uri = "http://127.0.0.1:19530"

        db_name = str(self._get_milvus_config("db_name", "default")).strip()
        user = str(self._get_milvus_config("user", "")).strip()
        password = str(self._get_milvus_config("password", "")).strip()
        token = str(self._get_milvus_config("token", "")).strip()

        self._client = MilvusClient(
            uri=uri,
            user=user,
            password=password,
            db_name=db_name,
            token=token,
        )

    async def _ensure_collection(self, dim: int | None = None):
        if self._collection_name is not None:
            return
        if not await self._ensure_client():
            return
        if not self._client:
            return
        async with self._lock:
            if self._collection_name is not None:
                return
            name = str(self._get_milvus_config("collection", "flomemo_working_memory"))
            if self._client.has_collection(name):
                self._client.load_collection(name)
                self._collection_name = name
                return
            if dim is None:
                return
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=64,
                ),
                FieldSchema(
                    name="session_id", dtype=DataType.VARCHAR, max_length=256
                ),
                FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=16),
                FieldSchema(name="ts", dtype=DataType.DOUBLE),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields, description="Flomemo working memory")
            index_params = IndexParams()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 8, "efConstruction": 64},
            )
            self._client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params,
            )
            self._client.load_collection(name)
            self._collection_name = name

    async def _prune(self):
        if not self._client or not self._collection_name:
            return
        retention_days = self._get_working_memory_int("retention_days", 3, minimum=1)
        cutoff = _now_ts() - retention_days * 86400
        expr = f"ts < {cutoff}"
        async with self._lock:
            await asyncio.to_thread(
                self._client.delete,
                self._collection_name,
                filter=expr,
            )

    def _format_search_results(self, results: Any) -> list[dict[str, Any]]:
        if not results:
            return []
        hits = results[0]
        formatted: list[dict[str, Any]] = []
        for hit in hits:
            if isinstance(hit, dict):
                entity = hit.get("entity", hit)
                distance = hit.get("distance")
            else:
                entity = getattr(hit, "entity", None)
                distance = getattr(hit, "distance", None)
            if not isinstance(entity, dict):
                continue
            score = 0.0
            if distance is not None:
                try:
                    score = 1 - float(distance)
                except (TypeError, ValueError):
                    score = 0.0
            formatted.append(
                {
                    "session_id": entity.get("session_id", ""),
                    "role": entity.get("role", ""),
                    "content": entity.get("content", ""),
                    "date": entity.get("date", ""),
                    "ts": entity.get("ts", 0.0),
                    "score": score,
                }
            )
        return formatted

    def _fallback_rank(
        self, query: str, rows: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            content = str(row.get("content", ""))
            score = _lexical_similarity(query, content)
            if score > 0:
                row["score"] = score
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

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

    def _get_config_int(self, key: str, default: int, minimum: int | None = None) -> int:
        value = self.config.get(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
        if minimum is not None and value < minimum:
            return minimum
        return value

    def _get_config_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _get_working_memory_config(self, key: str, default: Any) -> Any:
        container = self.config.get("working_memory", {})
        if isinstance(container, dict):
            return container.get(key, default)
        return default

    def _get_working_memory_int(
        self, key: str, default: int, minimum: int | None = None
    ) -> int:
        value = self._get_working_memory_config(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
        if minimum is not None and value < minimum:
            return minimum
        return value

    def _get_milvus_config(self, key: str, default: Any) -> Any:
        container = self.config.get("milvus", {})
        if isinstance(container, dict):
            return container.get(key, default)
        return default

    def _get_milvus_bool(self, key: str, default: bool) -> bool:
        value = self._get_milvus_config(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
