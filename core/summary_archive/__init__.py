import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context

if TYPE_CHECKING:
    from ..knowledge_graph import KnowledgeGraphStore
    from ..working_memory import WorkingMemoryStore

DEFAULT_SUMMARY_TIME = "23:50"
DEFAULT_SUMMARY_PROMPT = (
    "请基于以下对话内容生成一段 TL;DR，总结当天的核心信息。要求：\n"
    "1) 使用一段简洁自然语言，不分点；\n"
    "2) 强调人物、事件、时间、关键决定与结果；\n"
    "3) 保留关键数字/数量/规格；\n"
    "4) 不添加臆测与解释。\n\n"
    "对话内容：\n{content}"
)


def _now_ts() -> float:
    return time.time()


def _today_str(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.date().isoformat()


class SummaryArchive:
    def __init__(self, context: Context, config: AstrBotConfig, data_dir: Path):
        self.context = context
        self.config = config
        self._path = data_dir / "daily_summaries.json"
        self._lock = asyncio.Lock()
        self._summaries: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None

    async def load(self):
        async with self._lock:
            self._summaries = await self._load_json_file(self._path, [])

    async def save(self):
        async with self._lock:
            data = list(self._summaries)
        await self._save_json_file(self._path, data)

    def start_scheduler(
        self,
        working_memory: "WorkingMemoryStore",
        graph_store: "KnowledgeGraphStore | None",
    ) -> asyncio.Task:
        if self._task and not self._task.done():
            return self._task
        self._task = asyncio.create_task(
            self._summary_loop(working_memory, graph_store)
        )
        return self._task

    async def stop_scheduler(self):
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def run_daily_summary(
        self,
        working_memory: "WorkingMemoryStore",
        graph_store: "KnowledgeGraphStore | None",
    ):
        date_str = _today_str()
        entries = await working_memory.get_entries_for_date(date_str)
        existing = await self._existing_summary_sessions(date_str)
        if not entries:
            return

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in entries:
            session_id = item.get("session_id")
            if not session_id or session_id in existing:
                continue
            grouped.setdefault(session_id, []).append(item)

        min_messages = self._get_summary_int("summary_min_messages", 6, minimum=1)
        for session_id, items in grouped.items():
            items.sort(key=lambda x: x.get("ts", 0.0))
            if len(items) < min_messages:
                continue
            content_lines = [
                f"[{item.get('role', 'user')}] {item.get('content', '')}" for item in items
            ]
            conversation_text = "\n".join(content_lines)
            tldr = await self._generate_tldr(session_id, conversation_text)
            if not tldr:
                continue
            summary_item = {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "date": date_str,
                "tldr": tldr,
                "source_count": len(items),
                "created_at": _now_ts(),
            }
            async with self._lock:
                self._summaries.append(summary_item)
            await self.save()
            if graph_store and self._get_group_bool("graph", "graph_enabled", True):
                await graph_store.update_from_summary(session_id, date_str, tldr)

    async def get_recent(self, session_id: str, days: int) -> list[dict[str, Any]]:
        cutoff = datetime.now().date() - timedelta(days=days - 1)
        async with self._lock:
            items: list[dict[str, Any]] = []
            for item in self._summaries:
                if item.get("session_id") != session_id:
                    continue
                date_raw = item.get("date", "1970-01-01")
                try:
                    item_date = datetime.fromisoformat(date_raw).date()
                except ValueError:
                    logger.warning(f"摘要日期格式无效，已跳过: {date_raw}")
                    continue
                if item_date >= cutoff:
                    items.append(item)
        items.sort(key=lambda x: x.get("date", ""), reverse=True)
        return items

    async def get_for_date(self, session_id: str, date_str: str) -> list[dict[str, Any]]:
        async with self._lock:
            items = [
                item
                for item in self._summaries
                if item.get("session_id") == session_id and item.get("date") == date_str
            ]
        items.sort(key=lambda x: x.get("created_at", 0.0))
        return items

    async def reset_session(self, session_id: str):
        async with self._lock:
            self._summaries = [
                item for item in self._summaries if item.get("session_id") != session_id
            ]
        await self.save()

    async def count(self) -> int:
        async with self._lock:
            return len(self._summaries)

    async def _summary_loop(
        self,
        working_memory: "WorkingMemoryStore",
        graph_store: "KnowledgeGraphStore | None",
    ):
        while True:
            delay = self._seconds_to_next_summary()
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
            await self.run_daily_summary(working_memory, graph_store)

    def _seconds_to_next_summary(self) -> float:
        now = datetime.now()
        summary_time = self._parse_summary_time()
        target = now.replace(
            hour=summary_time.hour,
            minute=summary_time.minute,
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        return max((target - now).total_seconds(), 1.0)

    def _parse_summary_time(self) -> datetime:
        time_str = str(self._get_group_config("summary", "summary_time", DEFAULT_SUMMARY_TIME))
        match = re.match(r"^([01]\d|2[0-3]):([0-5]\d)$", time_str)
        if not match:
            logger.warning(f"summary_time 配置无效，回退为 {DEFAULT_SUMMARY_TIME}")
            match = re.match(r"^([01]\d|2[0-3]):([0-5]\d)$", DEFAULT_SUMMARY_TIME)
        if not match:
            return datetime.now()
        hour = int(match.group(1))
        minute = int(match.group(2))
        return datetime.now().replace(hour=hour, minute=minute)

    async def _generate_tldr(self, session_id: str, conversation: str) -> str:
        provider_id = await self._resolve_chat_provider_id(session_id)
        if not provider_id:
            logger.warning(f"无法获取 LLM provider，跳过 {session_id} 的摘要生成。")
            return ""
        prompt_template = self._get_group_config(
            "summary", "summary_prompt", DEFAULT_SUMMARY_PROMPT
        )
        prompt = str(prompt_template).format(content=conversation)
        llm_resp = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=prompt,
        )
        tldr = getattr(llm_resp, "completion_text", "").strip()
        return tldr

    async def _resolve_chat_provider_id(self, session_id: str) -> str | None:
        provider_id = str(self.config.get("llm_provider_id", "")).strip()
        if provider_id:
            return provider_id
        try:
            return await self.context.get_current_chat_provider_id(umo=session_id)
        except AttributeError:
            logger.warning("当前 AstrBot 版本不支持 get_current_chat_provider_id。")
            return None

    async def _existing_summary_sessions(self, date_str: str) -> set[str]:
        async with self._lock:
            return {
                item["session_id"]
                for item in self._summaries
                if item.get("date") == date_str and item.get("session_id")
            }

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

    def _get_config_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _get_group_config(self, group: str, key: str, default: Any) -> Any:
        container = self.config.get(group, {})
        if isinstance(container, dict):
            return container.get(key, default)
        return default

    def _get_group_bool(self, group: str, key: str, default: bool) -> bool:
        value = self._get_group_config(group, key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _get_summary_int(self, key: str, default: int, minimum: int | None = None) -> int:
        value = self._get_group_config("summary", key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
        if minimum is not None and value < minimum:
            return minimum
        return value
