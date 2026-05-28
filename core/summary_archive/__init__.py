import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context

if TYPE_CHECKING:
    from ..knowledge_graph import KnowledgeGraphStore
    from ..working_memory import WorkingMemoryStore

DEFAULT_SUMMARY_TIME = "23:50"
DEFAULT_SUMMARY_TIMEZONE = ""
DEFAULT_SAVE_DEBOUNCE_SECONDS = 2.0
DEFAULT_SUMMARY_PROMPT = (
    "请基于以下对话内容生成一段 TL;DR，总结当天的核心信息。要求：\n"
    "1) 使用一段简洁自然语言，不分点；\n"
    "2) 强调人物、事件、时间、关键决定与结果；\n"
    "3) 保留关键数字/数量/规格；\n"
    "4) 不添加臆测与解释。\n\n"
    "对话内容：\n{content}"
)


class SummaryArchive:
    def __init__(self, context: Context, config: AstrBotConfig, data_dir: Path):
        self.context = context
        self.config = config
        self._path = data_dir / "daily_summaries.json"
        self._state_path = data_dir / "daily_summaries_state.json"
        self._lock = asyncio.Lock()
        self._summaries: list[dict[str, Any]] = []
        self._state: dict[str, Any] = {
            "last_successful_date": "",
            "last_successful_run_at": 0.0,
            "timezone": "",
        }
        self._task: asyncio.Task | None = None
        self._save_task: asyncio.Task | None = None
        self._timezone: Any = None

    async def load(self):
        async with self._lock:
            self._summaries = await self._load_json_file(self._path, [])
            loaded_state = await self._load_json_file(self._state_path, {})
            if isinstance(loaded_state, dict):
                self._state.update(loaded_state)

    async def save(self):
        current = asyncio.current_task()
        if self._save_task and not self._save_task.done() and self._save_task is not current:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        self._save_task = None
        await self._persist_now()

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
            await self.save()
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        await self.save()

    async def run_daily_summary(
        self,
        working_memory: "WorkingMemoryStore",
        graph_store: "KnowledgeGraphStore | None",
        date_str: str | None = None,
    ) -> bool:
        date_str = date_str or self._today_str()
        entries = await working_memory.get_entries_for_date(date_str)
        existing = await self._existing_summary_sessions(date_str)
        if not entries:
            return True

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in entries:
            session_id = item.get("session_id")
            if not session_id or session_id in existing:
                continue
            grouped.setdefault(session_id, []).append(item)

        if not grouped:
            return True

        min_messages = self._get_summary_int("summary_min_messages", 6, minimum=1)
        min_turns = self._get_summary_int("summary_min_turns", 0, minimum=0)
        had_failures = False
        for session_id, items in grouped.items():
            items.sort(key=lambda x: x.get("ts", 0.0))
            source_message_count = sum(self._get_source_message_count(item) for item in items)
            source_turn_count = sum(self._get_source_turn_count(item) for item in items)
            if source_message_count < min_messages:
                continue
            if min_turns > 0 and source_turn_count < min_turns:
                continue
            content_lines = [
                self._format_working_memory_summary_line(item) for item in items
            ]
            conversation_text = "\n".join(content_lines)
            tldr = await self._generate_tldr(session_id, conversation_text)
            if tldr is None:
                had_failures = True
                continue
            if not tldr:
                continue
            summary_item = {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "date": date_str,
                "tldr": tldr,
                "source_summary_count": len(items),
                "source_message_count": source_message_count,
                "source_turn_count": source_turn_count,
                "created_at": time.time(),
            }
            async with self._lock:
                self._summaries.append(summary_item)
            self._request_save()
            if graph_store and self._get_group_bool("graph", "graph_enabled", True):
                await graph_store.update_from_summary(session_id, date_str, tldr)
        return not had_failures

    async def get_recent(self, session_id: str, days: int) -> list[dict[str, Any]]:
        cutoff = self._now().date() - timedelta(days=days - 1)
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
        await self._catch_up_missing_dates(working_memory, graph_store)
        while True:
            delay = self._seconds_to_next_summary()
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
            date_str = self._today_str()
            success = await self.run_daily_summary(working_memory, graph_store, date_str)
            if success:
                await self._mark_success(date_str)

    def _seconds_to_next_summary(self) -> float:
        now = self._now()
        hour, minute = self._parse_summary_time()
        target = now.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        return max((target - now).total_seconds(), 1.0)

    def _parse_summary_time(self) -> tuple[int, int]:
        time_str = str(self._get_group_config("summary", "summary_time", DEFAULT_SUMMARY_TIME))
        match = re.match(r"^([01]\d|2[0-3]):([0-5]\d)$", time_str)
        if not match:
            logger.warning(f"summary_time 配置无效，回退为 {DEFAULT_SUMMARY_TIME}")
            match = re.match(r"^([01]\d|2[0-3]):([0-5]\d)$", DEFAULT_SUMMARY_TIME)
        if not match:
            return 23, 50
        hour = int(match.group(1))
        minute = int(match.group(2))
        return hour, minute

    async def _generate_tldr(self, session_id: str, conversation: str) -> str | None:
        provider_id = await self._resolve_chat_provider_id(session_id)
        if not provider_id:
            logger.warning(f"无法获取 LLM provider，跳过 {session_id} 的摘要生成。")
            return None
        prompt_template = self._get_group_config(
            "summary", "summary_prompt", DEFAULT_SUMMARY_PROMPT
        )
        prompt = str(prompt_template).format(content=conversation)
        try:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
        except Exception as exc:
            logger.error(f"每日摘要生成失败: {exc}")
            return None
        return getattr(llm_resp, "completion_text", "").strip()

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

    def _request_save(self):
        if self._save_task and not self._save_task.done():
            return
        self._save_task = asyncio.create_task(self._delayed_save())

    async def _delayed_save(self):
        try:
            await asyncio.sleep(DEFAULT_SAVE_DEBOUNCE_SECONDS)
            await self._persist_now()
        except asyncio.CancelledError:
            pass
        finally:
            if asyncio.current_task() is self._save_task:
                self._save_task = None

    async def _persist_now(self):
        async with self._lock:
            data = list(self._summaries)
            state = dict(self._state)
        await self._save_json_file(self._path, data)
        await self._save_json_file(self._state_path, state)

    async def _catch_up_missing_dates(
        self,
        working_memory: "WorkingMemoryStore",
        graph_store: "KnowledgeGraphStore | None",
    ):
        available_dates = await working_memory.get_available_dates()
        if not available_dates:
            return
        target_date = self._latest_runnable_date()
        if target_date is None:
            return
        last_successful = self._parse_state_date(self._state.get("last_successful_date", ""))
        for date_str in available_dates:
            candidate = self._parse_state_date(date_str)
            if candidate is None or candidate > target_date:
                continue
            if last_successful is not None and candidate <= last_successful:
                continue
            success = await self.run_daily_summary(working_memory, graph_store, date_str)
            if not success:
                break
            await self._mark_success(date_str)

    async def _mark_success(self, date_str: str):
        async with self._lock:
            self._state["last_successful_date"] = date_str
            self._state["last_successful_run_at"] = time.time()
            self._state["timezone"] = self._get_timezone_name()
        self._request_save()

    def _latest_runnable_date(self):
        now = self._now()
        hour, minute = self._parse_summary_time()
        summary_time_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now >= summary_time_today:
            return now.date()
        return (now - timedelta(days=1)).date()

    def _parse_state_date(self, value: Any):
        try:
            return datetime.fromisoformat(str(value)).date()
        except (TypeError, ValueError):
            return None

    def _get_timezone(self):
        if self._timezone is not None:
            return self._timezone
        tz_name = self._get_timezone_name()
        if tz_name:
            try:
                self._timezone = ZoneInfo(tz_name)
                return self._timezone
            except ZoneInfoNotFoundError:
                logger.warning(f"summary_timezone 配置无效，回退到系统时区: {tz_name}")
        self._timezone = datetime.now().astimezone().tzinfo
        return self._timezone

    def _get_timezone_name(self) -> str:
        return str(self._get_group_config("summary", "summary_timezone", DEFAULT_SUMMARY_TIMEZONE)).strip()

    def _now(self) -> datetime:
        tz = self._get_timezone()
        return datetime.now(tz) if tz is not None else datetime.now()

    def _today_str(self) -> str:
        return self._now().date().isoformat()

    def _get_source_message_count(self, item: dict[str, Any]) -> int:
        value = item.get("source_count", 1)
        try:
            return max(int(value), 1)
        except (TypeError, ValueError):
            return 1

    def _get_source_turn_count(self, item: dict[str, Any]) -> int:
        value = item.get("source_turn_count", 0)
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return 0

    def _format_working_memory_summary_line(self, item: dict[str, Any]) -> str:
        message_count = self._get_source_message_count(item)
        turn_count = self._get_source_turn_count(item)
        counts = f"{message_count} messages"
        if turn_count > 0:
            counts += f", {turn_count} turns"
        return f"[working_memory_summary|{counts}] {item.get('content', '')}"

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
