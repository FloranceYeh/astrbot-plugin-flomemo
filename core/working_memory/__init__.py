import asyncio
import inspect
import json
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
        self._path = data_dir / "working_memory.json"
        self._lock = asyncio.Lock()
        self._records: list[dict[str, Any]] = []
        self._embedding_provider: Any = None
        self._client: MilvusClient | None = None
        self._collection_name: str | None = None
        self._connect_failed = False

    async def load(self):
        async with self._lock:
            self._records = await self._load_json_file(self._path, [])
        await self._ensure_collection()
        await self._prune()

    async def save(self):
        async with self._lock:
            records = list(self._records)
        await self._save_json_file(self._path, records)
        await self._ensure_collection()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        content = content.strip()
        if not content:
            return False
        record = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "date": _today_str(),
            "ts": _now_ts(),
        }
        if metadata:
            record.update(metadata)
        await self._append_local_record(record)

        embedding = await self._embed_text(content)
        if embedding is None:
            logger.warning("未生成 embedding，工作记忆仅写入本地文本索引。")
            await self._prune()
            return True
        await self._ensure_collection(dim=len(embedding))
        if not self._client or not self._collection_name:
            await self._prune()
            return True
        payload = [
            {
                **record,
                "embedding": embedding,
            }
        ]
        try:
            async with self._lock:
                await asyncio.to_thread(
                    self._client.insert,
                    self._collection_name,
                    payload,
                )
        except (MilvusException, OSError, ValueError) as exc:
            logger.warning(f"Milvus 写入失败，已保留本地文本记忆: {exc}")
        await self._prune()
        return True

    async def query(self, session_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        await self._ensure_collection()
        query_embedding = await self._embed_text(query)
        vector_hits = await self._query_vector_hits(session_id, query_embedding, top_k)
        local_rows = await self._get_local_session_rows(session_id)
        lexical_hits = self._fallback_rank(query, local_rows, top_k * 2)
        return self._merge_ranked_results(vector_hits, lexical_hits, top_k)

    async def get_entries_for_date(self, date_str: str) -> list[dict[str, Any]]:
        async with self._lock:
            rows = [
                dict(row) for row in self._records if str(row.get("date", "")) == date_str
            ]
        rows.sort(key=lambda x: x.get("ts", 0.0))
        return rows

    async def reset_session(self, session_id: str):
        async with self._lock:
            self._records = [
                item for item in self._records if item.get("session_id") != session_id
            ]
            records = list(self._records)
        await self._save_json_file(self._path, records)

        await self._ensure_collection()
        if not self._client or not self._collection_name:
            return
        expr = f'session_id == "{_escape_expr_value(session_id)}"'
        try:
            async with self._lock:
                await asyncio.to_thread(
                    self._client.delete,
                    self._collection_name,
                    filter=expr,
                )
        except (MilvusException, OSError, ValueError) as exc:
            logger.warning(f"Milvus 会话删除失败，本地文本记忆已清理: {exc}")

    async def count(self) -> int:
        async with self._lock:
            local_count = len(self._records)

        await self._ensure_collection()
        if not self._client or not self._collection_name:
            return local_count
        try:
            stats = await asyncio.to_thread(
                self._client.get_collection_stats, self._collection_name
            )
        except (MilvusException, OSError, ValueError):
            return local_count
        if not isinstance(stats, dict):
            return local_count
        count_value = stats.get("row_count", 0)
        try:
            return max(local_count, int(count_value))
        except (TypeError, ValueError):
            return local_count

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
        retention_days = self._get_working_memory_int("retention_days", 3, minimum=1)
        cutoff = _now_ts() - retention_days * 86400
        async with self._lock:
            kept_records = [
                item for item in self._records if float(item.get("ts", 0.0)) >= cutoff
            ]
            changed = len(kept_records) != len(self._records)
            self._records = kept_records
            records = list(self._records)
        if changed:
            await self._save_json_file(self._path, records)

        if not self._client or not self._collection_name:
            return
        expr = f"ts < {cutoff}"
        try:
            async with self._lock:
                await asyncio.to_thread(
                    self._client.delete,
                    self._collection_name,
                    filter=expr,
                )
        except (MilvusException, OSError, ValueError) as exc:
            logger.warning(f"Milvus 过期记忆清理失败: {exc}")

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
                    "id": entity.get("id", ""),
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
                enriched = dict(row)
                enriched["score"] = score
                scored.append((score, enriched))
        scored.sort(key=lambda x: (x[0], x[1].get("ts", 0.0)), reverse=True)
        return [item for _, item in scored[:top_k]]

    async def _query_vector_hits(
        self, session_id: str, query_embedding: list[float] | None, top_k: int
    ) -> list[dict[str, Any]]:
        if query_embedding is None or not self._client or not self._collection_name:
            return []
        expr = f'session_id == "{_escape_expr_value(session_id)}"'
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        try:
            async with self._lock:
                results = await asyncio.to_thread(
                    self._client.search,
                    self._collection_name,
                    data=[query_embedding],
                    filter=expr,
                    limit=top_k,
                    output_fields=["id", "session_id", "role", "content", "date", "ts"],
                    search_params=search_params,
                    anns_field="embedding",
                )
        except (MilvusException, OSError, ValueError) as exc:
            logger.warning(f"Milvus 检索失败，回退到本地文本记忆: {exc}")
            return []
        return self._format_search_results(results)

    async def _get_local_session_rows(self, session_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            rows = [
                dict(item) for item in self._records if item.get("session_id") == session_id
            ]
        return rows

    def _merge_ranked_results(
        self,
        vector_hits: list[dict[str, Any]],
        lexical_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in lexical_hits:
            key = str(item.get("id", "")) or str(uuid.uuid4())
            merged[key] = dict(item)
        for item in vector_hits:
            key = str(item.get("id", "")) or str(uuid.uuid4())
            current = merged.get(key, {})
            merged[key] = {
                **current,
                **item,
                "score": max(
                    float(current.get("score", 0.0)),
                    float(item.get("score", 0.0)),
                ),
            }

        ranked = sorted(
            merged.values(),
            key=lambda item: (
                float(item.get("score", 0.0)) + self._recency_bonus(item.get("ts", 0.0)),
                float(item.get("ts", 0.0)),
            ),
            reverse=True,
        )
        return ranked[:top_k]

    def _recency_bonus(self, ts: Any) -> float:
        try:
            age_seconds = max(_now_ts() - float(ts), 0.0)
        except (TypeError, ValueError):
            return 0.0
        retention_days = self._get_working_memory_int("retention_days", 3, minimum=1)
        horizon_seconds = retention_days * 86400
        if horizon_seconds <= 0:
            return 0.0
        freshness = 1.0 - min(age_seconds / horizon_seconds, 1.0)
        return freshness * 0.15

    async def _append_local_record(self, record: dict[str, Any]):
        async with self._lock:
            self._records.append(record)
            records = list(self._records)
        await self._save_json_file(self._path, records)

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
