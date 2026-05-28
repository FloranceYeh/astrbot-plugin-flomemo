from datetime import datetime

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools

from .core import KnowledgeGraphStore, SummaryArchive, WorkingMemoryStore


def _today_str() -> str:
    return datetime.now().date().isoformat()


class FlomenoMemory(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.data_dir = StarTools.get_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.working_memory = WorkingMemoryStore(context, config, self.data_dir)
        self.summary_archive = SummaryArchive(context, config, self.data_dir)
        self.knowledge_graph = KnowledgeGraphStore(context, config, self.data_dir)

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
        if user_text:
            await self.working_memory.add_message(session_id, "user", user_text)
        assistant_text = getattr(resp, "completion_text", None)
        if assistant_text:
            await self.working_memory.add_message(
                session_id, "assistant", str(assistant_text)
            )

    @filter.command_group("flomeno")
    def flomeno_group(self):
        """Flomeno 长期记忆指令组 /flomeno"""
        pass

    @flomeno_group.command("recall")  # type: ignore
    async def recall_cmd(self, event: AstrMessageEvent, query: str):
        """回忆相关记忆 /flomeno recall <query>"""
        session_id = event.unified_msg_origin
        if not session_id:
            yield event.plain_result("无法获取当前会话 ID。")
            return
        memory_block = await self._build_memory_block(session_id, query, include_graph=True)
        if not memory_block:
            yield event.plain_result("暂无可用记忆。")
            return
        yield event.plain_result(memory_block)

    @flomeno_group.command("tldr")  # type: ignore
    async def tldr_cmd(self, event: AstrMessageEvent, date: str | None = None):
        """查看指定日期摘要 /flomeno tldr [YYYY-MM-DD]"""
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

    @flomeno_group.command("graph")  # type: ignore
    async def graph_cmd(self, event: AstrMessageEvent, entity: str):
        """查询知识图谱 /flomeno graph <实体>"""
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

    @flomeno_group.command("status")  # type: ignore
    async def status_cmd(self, event: AstrMessageEvent):
        """查看记忆统计 /flomeno status"""
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

    @flomeno_group.command("reset")  # type: ignore
    async def reset_cmd(self, event: AstrMessageEvent, confirm: str | None = None):
        """清空当前会话记忆 /flomeno reset confirm"""
        if confirm != "confirm":
            yield event.plain_result("请使用 /flomeno reset confirm 确认清空当前会话记忆。")
            return
        session_id = event.unified_msg_origin
        if not session_id:
            yield event.plain_result("无法获取当前会话 ID。")
            return
        await self.working_memory.reset_session(session_id)
        await self.summary_archive.reset_session(session_id)
        yield event.plain_result("已清空当前会话的工作记忆与摘要记录。")

    async def terminate(self):
        await self.summary_archive.stop_scheduler()
        await self.working_memory.save()
        await self.summary_archive.save()
        await self.knowledge_graph.save()

    async def _build_memory_block(
        self, session_id: str, query: str, include_graph: bool = False
    ) -> str:
        top_k = self._get_config_int("working_memory_top_k", 5, minimum=1)
        working_items = await self.working_memory.query(session_id, query, top_k)
        summary_days = self._get_config_int("summary_injection_days", 7, minimum=1)
        summaries = await self.summary_archive.get_recent(session_id, summary_days)
        graph_edges: list[dict[str, str]] = []
        if include_graph or self._get_config_bool("graph_enabled", True):
            graph_edges = await self.knowledge_graph.query(query)

        sections: list[str] = []
        if working_items:
            sections.append("【工作记忆】")
            for item in working_items:
                sections.append(
                    f"- {item.get('date', '')} {item.get('role', '')}: {item.get('content', '')}"
                )
        if summaries:
            sections.append("【每日摘要】")
            for item in summaries:
                sections.append(f"- {item.get('date', '')}: {item.get('tldr', '')}")
        if graph_edges:
            max_edges = self._get_config_int("graph_max_edges", 6, minimum=0)
            trimmed = graph_edges[:max_edges] if max_edges > 0 else []
            if trimmed:
                sections.append("【知识图谱】")
                for edge in trimmed:
                    sections.append(
                        f"- {edge['source']} -[{edge['relation']}]-> {edge['target']}"
                    )
        return "\n".join(sections).strip()

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
