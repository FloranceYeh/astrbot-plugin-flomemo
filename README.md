# Flomemo Long-Term Memory

三层长期记忆插件：工作记忆（向量检索）、每日 TLDR 长期档案、知识图谱。

## 功能概览

- **工作记忆**：保存最近几天对话，支持向量检索回忆。
- **每日 TLDR**：每日定时将当天内容提炼成 TLDR 存档。
- **知识图谱**：抽取人物关系、事件因果等结构化关系。

## 指令

- `/flomemo recall <query>`：回忆相关记忆。
- `/flomemo tldr [YYYY-MM-DD]`：查看指定日期摘要。
- `/flomemo graph <实体>`：查询知识图谱关系。
- `/flomemo status`：查看记忆统计。
- `/flomemo reset confirm`：清空当前会话记忆。

## 配置

在 WebUI 中配置 `_conf_schema.json` 对应项，例如：

- `retention_days`：工作记忆保留天数。
- `summary_time`：每日 TLDR 生成时间（HH:MM）。
- `graph_enabled`：是否启用知识图谱抽取。
- `milvus.lite_path` / `milvus.address`：Milvus Lite 或远程地址配置。
- `milvus.collection`：工作记忆集合名称。

## 依赖

工作记忆使用 Milvus 向量数据库，请安装 `requirements.txt` 并配置 Milvus 连接信息。
