# CiteQuest API 契约

本文描述当前已实现的接口。启动服务后可通过 `/docs` 和 `/openapi.json` 查看字段定义；版本号由 `app/core/config.py` 的 `APP_VERSION` 同时提供给应用和 Python 包。

默认地址：`http://127.0.0.1:8000`。POST 请求使用 `Content-Type: application/json`。

| 接口 | 响应 | 用途 |
| --- | --- | --- |
| `GET /health` | JSON | 索引状态与可用检索能力 |
| `POST /search` | JSON | 论文检索，可选 AI Overview |
| `POST /ask` | JSON | 引用问答 |
| `POST /ask/stream` | SSE | 处理阶段与最终引用回答 |

## 索引状态

`GET /health` 返回 `status`、`version`、`paths`、`indexes`、`capabilities`。

- `status`：词法和向量索引均就绪为 `healthy`，仅词法索引就绪为 `degraded`，否则为 `unhealthy`。
- `indexes`：`metadata_db`、`fts5`、`faiss_index`、`faiss_id_map`、`faiss`。
- `capabilities`：`lexical_search`、`vector_search`、`hybrid_search`、`rag`。
- 路径来自运行时配置。检查的是本地索引可用性，不执行模型调用。

## 论文检索

`POST /search` 请求字段：

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `query` | string | 必填 | 非空查询 |
| `top_k` | integer | `10` | 1～100 条检索候选 |
| `mode` | string | `lexical` | `lexical`、`vector`、`hybrid` |
| `alpha` | number / null | `null` | Hybrid 词法权重，范围 0～1 |
| `year_from` / `year_to` | integer / null | `null` | 包含端点的发表年份范围 |
| `include_overview` | boolean | `false` | 由路由器判断是否附带生成回答 |

设置年份边界后，年份未知的论文会被排除；起始年份晚于结束年份返回 422。Hybrid 权重依次采用请求值、`CITEQUEST_HYBRID_ALPHA`、`0.5`；其他模式的 `effective_alpha` 为 `null`。

响应包含：

- `query`、`mode`、`effective_alpha`、`latency_ms`。
- `results`：按论文去重的结果列表，每项包含 `paper_id`、`chunk_id`、`title`、`year`、`venue`、`authors`、`score`、`snippet`、`abstract`。
- `total_results`：本次返回的论文数，不是整个语料库的匹配总数。去重后可能少于 `top_k`。
- `ai_overview`：引用回答或 `null`，结构同 `/ask`。
- `should_rag`、`rag_reason`：路由决定与原因；即使不请求 Overview 也会返回。
- `rewrite_keywords`：中文查询补充的英文检索词，未改写时为空字符串。

结果中的 `snippet` 是最多 300 字符的展示预览。RAG 从数据库读取对应文本块，不使用该预览充当完整证据。

页面和 `/search` 的 Overview 默认使用前 5 条候选。中文关键词改写最多生成 128 个输出 token，默认请求超时为 2 秒；调用 DeepSeek 官方接口时为该短任务设置 `thinking.type=disabled`，其他兼容接口不附加这一专属参数。失败或输出无效时回退到原查询。

## 引用问答

`POST /ask` 与 `/ask/stream` 使用相同请求结构：

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `question` | string | 必填 | 非空问题 |
| `top_k` | integer | `5` | 最多采用 1～20 条候选证据，可显式覆盖默认值 |
| `retrieval_mode` | string | `hybrid` | 独立检索时使用的模式 |
| `alpha` | number / null | `null` | Hybrid 权重，优先级同 `/search` |
| `pre_retrieved` | array / null | `null` | 可直接传入 `/search` 返回的结果对象 |

`pre_retrieved=null` 时执行检索；提供列表时复用这些候选，空列表也不会触发重新检索。候选首先按 `SearchResult` 结构转换，格式不完整的条目会跳过，再应用 `top_k`。

数据库中的文本块必须存在，且 `chunk_id` 与请求的 `paper_id` 归属一致。回答使用的正文、标题、年份、来源与 URL 均从 SQLite 读取。缺失、归属不符、空白或重复的文本块会被跳过，同一论文的不同文本块可以同时保留。引用只对应实际纳入上下文的证据，并从 `[1]` 连续编号。没有可用上下文时返回说明，不调用生成模型。

`POST /ask` 直接生成回答，不使用查询路由器跳过关键词问题。响应示例：

```json
{
  "question": "What is retrieval-augmented generation?",
  "answer": "The retrieved evidence describes ... [1]",
  "effective_alpha": 0.5,
  "citations": [
    {
      "citation_id": 1,
      "paper_id": "2501.00001",
      "chunk_id": "2501.00001_chunk0",
      "title": "Example paper",
      "url": "https://arxiv.org/abs/2501.00001"
    }
  ],
  "citation_valid": true,
  "citation_warnings": [],
  "latency_ms": 2345.0
}
```

`url` 允许为 `null`；不会为其他来源的论文拼接 arXiv 链接。`citation_valid` 检查回答中的引用编号是否对应上下文来源，不验证每条论断的语义正确性。无证据的说明响应保留 `citation_valid=true`，同时返回空引用列表。

## 流式问答

`POST /ask/stream` 返回 `text/event-stream`。这是阶段通知和完整回答的流式传送，未逐 token 推送模型输出。

每个事件使用 `event: 事件名` 与 `data: JSON内容` 两行，事件间用空行分隔。`done` 的 data 为空。

- 独立检索的阶段顺序：`retrieving → organizing → generating → verifying → result → done`。
- 复用已有候选时省略 `retrieving`；`result` 数据结构同 `/ask`。
- 没有可用上下文时：`organizing → result → done`，独立检索时前面还有 `retrieving`。
- 路由器判断无需生成时：`status` 的 `phase=skipped`，随后 `done`，不发送 `result`。
- 流建立后发生模型或处理异常时不会发送成功结果；客户端需处理流中断。

## 上下文预算

`CITEQUEST_RAG_CONTEXT_TOKENS` 为正整数，默认 `8000`，在构建上下文时读取。它只限制证据文本块，不包含系统提示、用户问题和生成输出。

计数采用无需额外下载的估算：4 个 ASCII 字符约 1 token，非 ASCII 字符保守按每字符 2 token 计入。不同模型的真实 tokenizer 计数可能不同，应为完整请求和回答保留余量。

通过来源校验的候选按检索顺序逐块加入；标题、年份、来源和分隔符也计入预算。若下一块的完整内容会超预算，就停止加入，不截断该块，也不用后面更短的块补位。第一块就超预算时返回无可用证据的说明，不调用生成模型。

日志 `build_evidence` 记录 `candidates`、`used`、`estimated_tokens`、`budget`、`budget_exhausted`。默认检索的 5 条是候选上限，最终采用条数还取决于来源校验、去重和预算。生成提示要求先说明核心证据缺口，不补写摘要未提供的数值、指标定义或实验条件；引用编号校验仍不等于语义支持校验。

## 错误响应

- 请求字段、枚举或年份范围不合法：422，响应使用 FastAPI 的 `detail` 错误列表。
- `/search` 缺少所需 FAISS 文件：503，`detail.error_code=INDEX_NOT_READY`。
- Hybrid 运行时权重配置错误：500，`detail.error_code=INVALID_HYBRID_ALPHA_CONFIGURATION`。
- 其他数据库、模型或配置异常没有统一的业务错误封装；普通请求可能返回 500，SSE 请求可能中断。

## RAG 评测口径

`python -m app.eval.rag_eval` 是命令行评测入口，不是 HTTP 接口。当前输出指标：

| 指标 | 定义 |
| --- | --- |
| `citation_precision` | 有效引用 ID 数 / 全部引用 ID 数；每个回答内部去重，不同回答分别计数；分母为零时为 `null` |
| `citation_validation_pass_rate` | `citation_valid=true` 的回答数 / 总回答数；包括无证据说明响应 |
| `no_citation_rate` | 没有引用标记的回答数 / 总回答数 |
| `avg_citations_per_answer` | 每个回答使用的去重引用 ID 数均值 |
| `avg_latency_ms` | 问答总耗时均值 |

汇总中的 `total_citations`、`valid_citations` 与逐题的有效/无效引用计数可用于复核。旧脚本把整条回答的校验通过率写在 `citation_precision` 字段中；旧报告的这个字段不能直接与修正后的精确率比较。
