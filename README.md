# Flomemo Long-Term Memory

三层长期记忆插件：工作记忆（向量检索）、每日 TLDR 长期档案、知识图谱。

## 功能概览

- **工作记忆**：按配置的对话轮数生成摘要并保存，优先走向量检索；没有 Embedding 时会回退到本地文本记忆。
- **每日 TLDR**：每日定时基于当天的工作记忆摘要做二级聚合，生成 TLDR 存档。
- **知识图谱**：抽取人物关系、事件因果等结构化关系。

## 指令

- `/flomemo recall <query>`：回忆相关记忆。
- `/flomemo tldr [YYYY-MM-DD]`：查看指定日期摘要。
- `/flomemo graph <实体>`：查询知识图谱关系。
- `/flomemo status`：查看记忆统计。
- `/flomemo reset confirm`：清空当前会话记忆。

## 配置

在 WebUI 中配置 `_conf_schema.json` 对应项，例如：

- `working_memory.retention_days`：工作记忆保留天数。
- `working_memory.working_memory_batch_size`：每 N 轮对话生成一次工作记忆摘要。
- `working_memory.working_memory_summary_prompt`：工作记忆摘要提示词。
- `summary.summary_time`：每日 TLDR 生成时间（HH:MM）。
- `summary.summary_min_messages`：生成 TLDR 所需的最小原始消息条数。
- `summary.summary_min_turns`：生成 TLDR 所需的最小原始对话轮数，`0` 表示不限制。
- `summary.summary_injection_days`：注入到会话中的摘要回看天数。
- `graph.graph_enabled`：是否启用知识图谱抽取。
- `graph.graph_max_edges`：注入到会话中的图谱边数量上限。
- `memory_injection_max_chars`：注入记忆的总字符预算，超出时自动裁剪。
- `milvus.lite_path` / `milvus.address`：Milvus Lite 或远程地址配置。
- `milvus.collection`：工作记忆集合名称。

除根级配置 `embedding_provider_id`、`llm_provider_id`、`memory_injection`、`memory_injection_target` 外，其余配置建议按对象分组填写。

```json
{
  "embedding_provider_id": "",
  "llm_provider_id": "",
  "memory_injection": true,
  "memory_injection_max_chars": 1800,
  "memory_injection_target": "system_prompt",
  "working_memory": {
    "retention_days": 3,
    "working_memory_top_k": 5,
    "working_memory_batch_size": 5,
    "working_memory_summary_prompt": "请将以下对话内容压缩成一段可检索的工作记忆摘要。要求：\\n1) 保留关键人物、事件、时间、结论与约束；\\n2) 保留重要数字/数量/规格；\\n3) 使用简洁自然语言，不分点；\\n4) 不引入臆测与解释。\\n\\n对话内容：\\n{content}"
  },
  "summary": {
    "summary_time": "23:50",
    "summary_min_messages": 6,
    "summary_min_turns": 0,
    "summary_prompt": "请基于以下对话内容生成一段 TL;DR，总结当天的核心信息。要求：\\n1) 使用一段简洁自然语言，不分点；\\n2) 强调人物、事件、时间、关键决定与结果；\\n3) 保留关键数字/数量/规格；\\n4) 不添加臆测与解释。\\n\\n对话内容：\\n{content}",
    "summary_injection_days": 7
  },
  "graph": {
    "graph_enabled": true,
    "graph_max_edges": 6,
    "graph_prompt": "请从以下 TL;DR 中抽取人物关系、事件因果与关键事实，输出 JSON 数组。每个元素包含：source, relation, target, type, evidence。\\n仅输出 JSON，不要额外说明。\\n\\nTL;DR：\\n{summary}"
  },
  "milvus": {
    "lite_path": "milvus/flomemo.db",
    "db_name": "default",
    "collection": "flomemo_working_memory"
  }
}
```

## Milvus 配置示例

在 AstrBot 的插件配置界面，填写 `milvus` 对象即可：

```json
{
  "milvus": {
    "lite_path": "milvus/flomemo.db",
    "db_name": "default",
    "collection": "flomemo_working_memory"
  }
}
```

未配置 `milvus.lite_path` 且未填写 `milvus.address` 时，默认会创建到：
`data\plugin_data\astrbot_plugin_flomemo\milvus\flomemo.db`。

如需连接远程 Milvus（Standalone 或托管服务）：

```json
{
  "milvus": {
    "address": "127.0.0.1:19530",
    "db_name": "default",
    "user": "",
    "password": "",
    "token": "",
    "secure": false,
    "collection": "flomemo_working_memory"
  }
}
```

## 依赖

工作记忆优先使用 Milvus 做向量检索，请安装 `requirements.txt` 并配置 Milvus 连接信息。即使没有可用的 Embedding Provider，插件也会把摘要保留到本地文本记忆中。

## 摘要链路

- 工作记忆层先按 `working_memory.working_memory_batch_size` 把多轮对话压成摘要，并为每条摘要保存 `source_count` 与 `source_turn_count` 元数据。
- 每日 TLDR 层不会直接按“摘要条数”判断是否生成，而是先累计这些元数据，再按 `summary.summary_min_messages` / `summary.summary_min_turns` 判断是否满足阈值。
- 因此每日 TLDR 依然是基于工作记忆摘要做二级聚合，但阈值统计使用的是真实原始消息量和对话轮数。
