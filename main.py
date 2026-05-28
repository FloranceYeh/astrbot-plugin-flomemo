from datetime import datetime

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools

from .core import KnowledgeGraphStore, SummaryArchive, WorkingMemoryStore

DEFAULT_WORKING_MEMORY_PROMPT = (
    "请将以下对话内容压缩成一段可检索的工作记忆摘要。要求：\n"
    "1) 保留关键人物、事件、时间、结论与约束；\n"
    "2) 保留重要数字/数量/规格；\n"
    "3) 使用简洁自然语言，不分点；\n"
    "4) 不引入臆测与解释。\n\n"
    "对话内容：\n{content}"
)
DEFAULT_MEMORY_INJECTION_MAX_CHARS = 1800


def _today_str() -> str:
    return datetime.now().date().isoformat()


class FlomemoMemory(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.data_dir = StarTools.get_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.working_memory = WorkingMemoryStore(context, config, self.data_dir)
        self.summary_archive = SummaryArchive(context, config, self.data_dir)
        self.knowledge_graph = KnowledgeGraphStore(context, config, self.data_dir)
        self._working_memory_buffer: dict[str, list[dict[str, str]]] = {}
        self._working_memory_turns: dict[str, int] = {}

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        await self.working_memory.load()
        await self.summary_archive.load()
        await self.knowledge_graph.load()
        self.summary_archive.start_scheduler(self.working_memory, self.knowledge_graph)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._get_config_bool("memory_injection", True):
            return
        query_text = ""
        if isinstance(req.prompt, str) and req.prompt.strip():
            query_text = req.prompt
        elif event.message_str:
            query_text = event.message_str
        if not query_text:
            return
        session_id = event.unified_msg_origin
        if not session_id:
            return
        memory_block = await self._build_memory_block(session_id, query_text)
        if not memory_block:
            return
        target = self.config.get("memory_injection_target", "system_prompt")
        if target == "user_prompt":
            current = req.prompt if isinstance(req.prompt, str) else ""
            req.prompt = f"{memory_block}\n\n{current}".strip()
        else:
            current = req.system_prompt if isinstance(req.system_prompt, str) else ""
            req.system_prompt = f"{current}\n\n{memory_block}".strip()

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        session_id = event.unified_msg_origin
        if not session_id:
            return
        user_text = event.message_str
        assistant_text = getattr(resp, "completion_text", None)
        await self._append_working_memory_batch(
            session_id,
            user_text,
            str(assistant_text) if assistant_text else None,
        )

    @filter.command_group("flomemo")
    def flomemo_group(self):
        """Flomemo 长期记忆指令组 /flomemo"""
        pass

    @flomemo_group.command("recall")  # type: ignore
    async def recall_cmd(self, event: AstrMessageEvent, query: str):
        """回忆相关记忆 /flomemo recall <query>"""
        session_id = event.unified_msg_origin
        if not session_id:
            yield event.plain_result("无法获取当前会话 ID。")
            return
        memory_block = await self._build_memory_block(session_id, query, include_graph=True)
        if not memory_block:
            yield event.plain_result("暂无可用记忆。")
            return
        yield event.plain_result(memory_block)

    @flomemo_group.command("tldr")  # type: ignore
    async def tldr_cmd(self, event: AstrMessageEvent, date: str | None = None):
        """查看指定日期摘要 /flomemo tldr [YYYY-MM-DD]"""
        session_id = event.unified_msg_origin
        if not session_id:
            yield event.plain_result("无法获取当前会话 ID。")
            return
        target_date = date or _today_str()
        summaries = await self.summary_archive.get_for_date(session_id, target_date)
        if not summaries:
            yield event.plain_result("暂无摘要记录。")
            return
        lines = [f"{item['date']}: {item['tldr']}" for item in summaries]
        yield event.plain_result("\n".join(lines))

    @flomemo_group.command("graph")  # type: ignore
    async def graph_cmd(self, event: AstrMessageEvent, entity: str):
        """查询知识图谱 /flomemo graph <实体>"""
        edges = await self.knowledge_graph.query(entity)
        if not edges:
            yield event.plain_result("暂无相关图谱关系。")
            return
        lines = [
            f"{edge['source']} -[{edge['relation']}]-> {edge['target']}"
            + (f" ({edge['date']})" if edge.get("date") else "")
            for edge in edges
        ]
        yield event.plain_result("\n".join(lines))

    @flomemo_group.command("status")  # type: ignore
    async def status_cmd(self, event: AstrMessageEvent):
        """查看记忆统计 /flomemo status"""
        working_count = await self.working_memory.count()
        summary_count = await self.summary_archive.count()
        graph_nodes = await self.knowledge_graph.count_nodes()
        graph_edges = await self.knowledge_graph.count_edges()
        yield event.plain_result(
            "记忆统计:\n"
            f"- 工作记忆: {working_count}\n"
            f"- 每日摘要: {summary_count}\n"
            f"- 图谱节点: {graph_nodes}\n"
            f"- 图谱边: {graph_edges}"
        )

    @flomemo_group.command("reset")  # type: ignore
    async def reset_cmd(self, event: AstrMessageEvent, confirm: str | None = None):
        """清空当前会话记忆 /flomemo reset confirm"""
        if confirm != "confirm":
            yield event.plain_result("请使用 /flomemo reset confirm 确认清空当前会话记忆。")
            return
        session_id = event.unified_msg_origin
        if not session_id:
            yield event.plain_result("无法获取当前会话 ID。")
            return
        await self.working_memory.reset_session(session_id)
        await self.summary_archive.reset_session(session_id)
        self._working_memory_buffer.pop(session_id, None)
        self._working_memory_turns.pop(session_id, None)
        yield event.plain_result("已清空当前会话的工作记忆与摘要记录。")

    async def terminate(self):
        await self.summary_archive.stop_scheduler()
        await self.working_memory.save()
        await self.summary_archive.save()
        await self.knowledge_graph.save()
        self._working_memory_buffer.clear()
        self._working_memory_turns.clear()

    async def _build_memory_block(
        self, session_id: str, query: str, include_graph: bool = False
    ) -> str:
        top_k = self._get_group_int("working_memory", "working_memory_top_k", 5, minimum=1)
        working_items = await self.working_memory.query(session_id, query, top_k)
        summary_days = self._get_group_int("summary", "summary_injection_days", 7, minimum=1)
        summaries = await self.summary_archive.get_recent(session_id, summary_days)
        graph_edges: list[dict[str, str]] = []
        if include_graph or self._get_group_bool("graph", "graph_enabled", True):
            graph_edges = await self.knowledge_graph.query(query)

        remaining = self._get_config_int(
            "memory_injection_max_chars",
            DEFAULT_MEMORY_INJECTION_MAX_CHARS,
            minimum=300,
        )
        sections: list[str] = []
        seen: set[str] = set()
        if working_items:
            lines = [
                f"- {item.get('date', '')} {item.get('role', '')}: {item.get('content', '')}"
                for item in working_items
            ]
            remaining = self._append_memory_section(
                "【工作记忆】", lines, sections, seen, remaining
            )
        if summaries:
            lines = [f"- {item.get('date', '')}: {item.get('tldr', '')}" for item in summaries]
            remaining = self._append_memory_section(
                "【每日摘要】", lines, sections, seen, remaining
            )
        if graph_edges:
            graph_edges.sort(key=lambda edge: str(edge.get("date", "")), reverse=True)
            max_edges = self._get_group_int("graph", "graph_max_edges", 6, minimum=0)
            trimmed = graph_edges[:max_edges] if max_edges > 0 else []
            if trimmed:
                lines = [
                    f"- {edge['source']} -[{edge['relation']}]-> {edge['target']}"
                    for edge in trimmed
                ]
                remaining = self._append_memory_section(
                    "【知识图谱】", lines, sections, seen, remaining
                )
        return "\n\n".join(sections).strip()

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

    def _get_group_config(self, group: str, key: str, default: object) -> object:
        container = self.config.get(group, {})
        if isinstance(container, dict):
            return container.get(key, default)
        return default

    def _get_group_int(
        self,
        group: str,
        key: str,
        default: int,
        minimum: int | None = None,
    ) -> int:
        value = self._get_group_config(group, key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
        if minimum is not None and value < minimum:
            return minimum
        return value

    def _get_group_bool(self, group: str, key: str, default: bool) -> bool:
        value = self._get_group_config(group, key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _append_memory_section(
        self,
        title: str,
        lines: list[str],
        sections: list[str],
        seen: set[str],
        remaining: int,
    ) -> int:
        if remaining <= len(title) + 4:
            return remaining
        kept_lines: list[str] = []
        current_length = len(title)
        for line in lines:
            normalized = self._normalize_memory_line(line)
            if not normalized or normalized in seen:
                continue
            next_length = current_length + 1 + len(line)
            if kept_lines and next_length > remaining:
                break
            if not kept_lines and next_length > remaining:
                line = self._truncate_memory_line(line, remaining - len(title) - 1)
                if not line:
                    break
                normalized = self._normalize_memory_line(line)
            kept_lines.append(line)
            seen.add(normalized)
            current_length = len(title) + sum(len(item) + 1 for item in kept_lines)
            if current_length >= remaining:
                break
        if not kept_lines:
            return remaining
        block = "\n".join([title, *kept_lines])
        sections.append(block)
        return max(remaining - len(block) - 2, 0)

    def _normalize_memory_line(self, line: str) -> str:
        return " ".join(line.strip().lower().split())

    def _truncate_memory_line(self, line: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        if len(line) <= max_length:
            return line
        if max_length <= 3:
            return line[:max_length]
        return f"{line[: max_length - 3].rstrip()}..."

    async def _append_working_memory_batch(
        self,
        session_id: str,
        user_text: str | None,
        assistant_text: str | None,
    ):
        if not user_text and not assistant_text:
            return
        batch_size = self._get_group_int(
            "working_memory", "working_memory_batch_size", 5, minimum=1
        )
        messages = self._working_memory_buffer.setdefault(session_id, [])
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
        self._working_memory_turns[session_id] = (
            self._working_memory_turns.get(session_id, 0) + 1
        )
        if self._working_memory_turns[session_id] < batch_size:
            return
        summary = await self._summarize_working_memory(session_id, messages)
        if not summary:
            return
        stored = await self.working_memory.add_message(session_id, "summary", summary)
        if not stored:
            return
        self._working_memory_buffer[session_id] = []
        self._working_memory_turns[session_id] = 0

    async def _summarize_working_memory(
        self, session_id: str, messages: list[dict[str, str]]
    ) -> str:
        provider_id = await self._resolve_llm_provider_id(session_id)
        if not provider_id:
            return ""
        content_lines = [
            f"[{item.get('role', 'user')}] {item.get('content', '')}"
            for item in messages
            if item.get("content")
        ]
        if not content_lines:
            return ""
        prompt_template = self._get_group_config(
            "working_memory",
            "working_memory_summary_prompt",
            DEFAULT_WORKING_MEMORY_PROMPT,
        )
        prompt = str(prompt_template).format(content="\n".join(content_lines))
        try:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
        except Exception as exc:
            logger.error(f"工作记忆摘要生成失败: {exc}")
            return ""
        return getattr(llm_resp, "completion_text", "").strip()

    async def _resolve_llm_provider_id(self, session_id: str) -> str | None:
        provider_id = str(self.config.get("llm_provider_id", "")).strip()
        if provider_id:
            return provider_id
        try:
            return await self.context.get_current_chat_provider_id(umo=session_id)
        except AttributeError:
            return None
