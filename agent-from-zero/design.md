# design.md

# 从 0 开始搭建 Agent 教程设计文档

> 教程主题：基于 LangChain + LangGraph + LangSmith，从 0 到 1 搭建一个可观测、可评估、可扩展的 Agent 系统。  
> 使用方式：本文件作为 Codex / AI 编程助手的顶层设计文档。后续可根据每一章的 `目标`、`产出`、`任务拆分` 和 `验收标准` 自动生成完整教程、代码、README 和示例项目。

---

## 1. 教程定位

### 1.1 教程目标

本教程面向希望从零开始理解并搭建 Agent 系统的开发者，最终目标不是只做一个聊天机器人，而是完成一个具备以下能力的工程化 Agent 项目：

1. 能调用大模型完成基础对话和结构化输出。
2. 能接入外部工具，例如搜索、计算、文件读取、数据库查询。
3. 能基于本地文档构建 RAG 知识库。
4. 能用 LangGraph 编排多步骤 Agent 工作流。
5. 能实现状态保存、错误恢复、人机协同审核。
6. 能使用 LangSmith 进行 tracing、debug、evaluation 和实验对比。
7. 能最终部署成一个可演示的 Web API 或简单 Web App。

### 1.2 目标读者

适合以下读者：

- 有 Python 基础，但没有系统学过 LangChain / LangGraph。
- 想搭建自己的 Agent 应用。
- 想做 RAG、工具调用、多 Agent、自动报告生成等任务。
- 想把大模型 Demo 进一步升级为可维护的工程项目。
- 想使用 Codex 自动生成每一章的教程和代码。

### 1.3 前置知识

读者需要具备：

```text
Python 基础语法
命令行基础
pip / uv / conda 基础
JSON 基础
HTTP API 基础概念
Git 基础
```

不强制要求：

```text
复杂后端经验
深度学习训练经验
云原生经验
前端开发经验
```

---

## 2. 技术栈设计

### 2.1 核心框架

```text
Python 3.11+
LangChain
LangGraph
LangSmith
langchain-openai
langchain-community
python-dotenv
pydantic
```

### 2.2 可选模型供应商

教程默认使用 OpenAI 接口，但应在设计中保留模型可替换能力。

可支持：

```text
OpenAI
DeepSeek
Qwen
Claude
Azure OpenAI
本地模型，例如 Ollama
```

### 2.3 RAG 相关组件

基础版本：

```text
FAISS 或 Chroma
本地 PDF / TXT / Markdown 文档
RecursiveCharacterTextSplitter
OpenAI Embeddings 或兼容 Embeddings
```

进阶版本：

```text
Qdrant
Milvus
PostgreSQL + pgvector
Hybrid Search
Reranker
Metadata Filter
```

### 2.4 工程化组件

基础版本：

```text
FastAPI
Streamlit
SQLite
.env 环境变量
pytest
```

进阶版本：

```text
PostgreSQL
Redis
Celery / RQ
Docker
Docker Compose
CI/CD
LangSmith Deployment 或自建服务
```

---

## 3. 最终项目设计

### 3.1 项目名称

```text
agent-from-zero
```

### 3.2 项目主题

构建一个“资料分析 Agent”。

用户可以输入问题、上传文档，Agent 能够：

1. 回答普通问题。
2. 检索本地知识库。
3. 调用工具。
4. 按流程完成复杂任务。
5. 输出结构化结果。
6. 在关键步骤请求人工确认。
7. 在 LangSmith 中查看完整运行轨迹。
8. 用测试集评估回答质量。

### 3.3 最终能力

最终项目需要具备：

```text
基础 Chat
结构化 JSON 输出
工具调用
RAG 文档问答
LangGraph 状态机
条件分支
循环重试
人工审核节点
持久化 checkpoint
LangSmith tracing
LangSmith evaluation
FastAPI 接口
Streamlit Demo 页面
```

### 3.4 示例业务场景

教程可以使用一个通用场景：研究资料分析助手。

输入：

```text
用户上传若干 PDF / Markdown / TXT 文档
用户提出问题
用户要求 Agent 生成摘要、表格、建议或报告
```

输出：

```text
回答
引用来源
结构化 JSON
Markdown 报告
人工审核后的最终版本
```

---

## 4. 推荐仓库结构

```text
agent-from-zero/
│
├── README.md
├── design.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── docs/
│   ├── chapter_01_environment.md
│   ├── chapter_02_llm_basics.md
│   ├── chapter_03_langchain_basics.md
│   ├── chapter_04_tool_calling.md
│   ├── chapter_05_rag_basics.md
│   ├── chapter_06_langgraph_basics.md
│   ├── chapter_07_langgraph_advanced.md
│   ├── chapter_08_langsmith_tracing.md
│   ├── chapter_09_langsmith_evaluation.md
│   ├── chapter_10_api_and_ui.md
│   └── chapter_11_production.md
│
├── src/
│   └── agent_from_zero/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── settings.py
│       │   └── logging_config.py
│       │
│       ├── models/
│       │   ├── llm_factory.py
│       │   └── schemas.py
│       │
│       ├── chains/
│       │   ├── simple_chat.py
│       │   ├── structured_output.py
│       │   └── summarizer.py
│       │
│       ├── tools/
│       │   ├── calculator.py
│       │   ├── web_search.py
│       │   ├── file_reader.py
│       │   └── report_writer.py
│       │
│       ├── rag/
│       │   ├── loaders.py
│       │   ├── splitters.py
│       │   ├── vectorstore.py
│       │   ├── retriever.py
│       │   └── qa_chain.py
│       │
│       ├── graphs/
│       │   ├── state.py
│       │   ├── basic_graph.py
│       │   ├── rag_graph.py
│       │   ├── tool_graph.py
│       │   ├── human_review_graph.py
│       │   └── multi_agent_graph.py
│       │
│       ├── evaluation/
│       │   ├── datasets.py
│       │   ├── evaluators.py
│       │   └── run_eval.py
│       │
│       ├── api/
│       │   ├── main.py
│       │   ├── routers.py
│       │   └── schemas.py
│       │
│       └── ui/
│           └── streamlit_app.py
│
├── examples/
│   ├── 01_simple_llm.py
│   ├── 02_structured_output.py
│   ├── 03_tool_calling.py
│   ├── 04_rag_qa.py
│   ├── 05_basic_langgraph.py
│   ├── 06_human_in_loop.py
│   ├── 07_langsmith_tracing.py
│   └── 08_evaluation.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── vectorstore/
│
├── tests/
│   ├── test_chains.py
│   ├── test_tools.py
│   ├── test_rag.py
│   └── test_graphs.py
│
└── scripts/
    ├── ingest_docs.py
    ├── run_graph.py
    ├── run_api.py
    └── run_ui.py
```

---

## 5. 教程章节设计

---

# Chapter 01：环境搭建

## 目标

让读者完成项目初始化，并能成功调用一次大模型。

## 内容

1. 创建项目目录。
2. 创建虚拟环境。
3. 安装依赖。
4. 配置 `.env`。
5. 编写第一个 LLM 调用脚本。
6. 验证 LangSmith 环境变量是否生效。

## 推荐依赖

```bash
uv init agent-from-zero
uv add langchain langchain-openai langgraph langsmith python-dotenv pydantic
uv add langchain-community faiss-cpu chromadb
uv add fastapi uvicorn streamlit pytest
```

## 产出文件

```text
.env.example
src/agent_from_zero/config/settings.py
examples/01_simple_llm.py
README.md
```

## 验收标准

运行：

```bash
python examples/01_simple_llm.py
```

能够输出：

```text
Hello, Agent!
```

或模型生成的简单回答。

---

# Chapter 02：不使用框架，理解 LLM 调用基础

## 目标

先理解大模型调用的底层结构，再进入 LangChain。

## 内容

1. messages 格式。
2. system / user / assistant 消息。
3. temperature。
4. streaming。
5. JSON 输出。
6. 错误处理。
7. 重试机制。

## 产出文件

```text
examples/02_raw_llm_call.py
src/agent_from_zero/models/llm_factory.py
```

## 核心示例

输入一段资料，让模型输出 JSON：

```json
{
  "title": "",
  "summary": "",
  "keywords": [],
  "action_items": []
}
```

## 验收标准

模型输出可以被 `json.loads()` 或 Pydantic 正确解析。

---

# Chapter 03：LangChain 基础

## 目标

掌握 LangChain 的基本抽象。

## 内容

1. ChatModel。
2. PromptTemplate / ChatPromptTemplate。
3. OutputParser。
4. Runnable。
5. RunnableSequence。
6. LCEL 基础。
7. Pydantic 结构化输出。

## 产出文件

```text
src/agent_from_zero/chains/simple_chat.py
src/agent_from_zero/chains/structured_output.py
src/agent_from_zero/models/schemas.py
examples/02_structured_output.py
```

## 练习

构建一个“资料摘要链”：

输入：

```text
一段研究资料
```

输出：

```json
{
  "topic": "",
  "summary": "",
  "key_points": [],
  "limitations": [],
  "next_steps": []
}
```

## 验收标准

1. 能正常运行 chain。
2. 输出格式稳定。
3. 错误输入时有合理异常提示。

---

# Chapter 04：Tool Calling 基础

## 目标

让 Agent 能调用外部函数。

## 内容

1. 什么是 tool。
2. 如何定义 Python 函数工具。
3. 如何为工具添加描述。
4. 如何绑定工具到模型。
5. 如何解析 tool call。
6. 工具调用失败时如何处理。
7. 工具调用安全边界。

## 产出文件

```text
src/agent_from_zero/tools/calculator.py
src/agent_from_zero/tools/file_reader.py
src/agent_from_zero/tools/report_writer.py
examples/03_tool_calling.py
```

## 示例工具

```text
calculator：执行简单数学计算
read_text_file：读取本地文本
write_markdown_report：写入 Markdown 报告
```

## 验收标准

用户输入：

```text
请计算 18 * 27，并写入报告。
```

Agent 能：

1. 调用 calculator。
2. 得到结果。
3. 调用 report_writer。
4. 生成 Markdown 文件。

---

# Chapter 05：RAG 文档问答

## 目标

构建一个本地知识库问答系统。

## 内容

1. 文档加载。
2. 文本切分。
3. Embedding。
4. 向量数据库。
5. Retriever。
6. RAG Prompt。
7. 引用来源。
8. 召回结果调试。

## 产出文件

```text
src/agent_from_zero/rag/loaders.py
src/agent_from_zero/rag/splitters.py
src/agent_from_zero/rag/vectorstore.py
src/agent_from_zero/rag/retriever.py
src/agent_from_zero/rag/qa_chain.py
scripts/ingest_docs.py
examples/04_rag_qa.py
```

## 示例流程

```text
读取 data/raw 中的文档
切分成 chunks
写入 FAISS / Chroma
用户提问
检索相关 chunks
模型生成答案
附带引用来源
```

## 验收标准

用户提问：

```text
这批文档主要讨论了什么？
```

系统返回：

```text
答案
引用的文档名
引用的 chunk 编号
```

---

# Chapter 06：LangGraph 基础

## 目标

用 LangGraph 把单步 chain 改造成多步骤工作流。

## 内容

1. State。
2. Node。
3. Edge。
4. START / END。
5. Conditional Edge。
6. Graph 编译。
7. Graph 运行。
8. State 更新规则。

## 产出文件

```text
src/agent_from_zero/graphs/state.py
src/agent_from_zero/graphs/basic_graph.py
examples/05_basic_langgraph.py
```

## 示例工作流

```text
用户输入
↓
意图识别节点
↓
普通问答 / RAG 问答 / 工具调用
↓
答案生成
↓
结束
```

## 验收标准

对于不同输入，Graph 能自动走不同分支：

```text
普通问题 → chat node
文档问题 → rag node
计算问题 → tool node
```

---

# Chapter 07：LangGraph 进阶：循环、重试与人机协同

## 目标

构建更接近真实业务的 Agent 工作流。

## 内容

1. 结果质量检查节点。
2. 不合格时自动重试。
3. 人工审核节点。
4. interrupt。
5. checkpoint。
6. thread_id。
7. 从中断状态恢复。
8. 错误处理分支。

## 产出文件

```text
src/agent_from_zero/graphs/human_review_graph.py
examples/06_human_in_loop.py
```

## 示例工作流

```text
资料输入
↓
生成初版报告
↓
质量检查
↓
如果质量不合格：返回修改
↓
如果需要人工审核：暂停
↓
人工确认
↓
生成最终报告
```

## 验收标准

1. Graph 可以暂停在人工审核节点。
2. 用户输入 approve / revise 后继续运行。
3. 重新运行时不会丢失之前状态。

---

# Chapter 08：多 Agent 工作流

## 目标

用 LangGraph 构建多个角色协作的 Agent 系统。

## 内容

1. 多 Agent 的适用场景。
2. Supervisor 模式。
3. Specialist Agent 模式。
4. Subgraph。
5. Agent 间状态传递。
6. 防止无限循环。
7. 防止角色职责混乱。

## 产出文件

```text
src/agent_from_zero/graphs/multi_agent_graph.py
examples/07_multi_agent.py
```

## 示例角色

```text
Planner Agent：拆解任务
Research Agent：检索资料
Tool Agent：调用工具
Writer Agent：生成报告
Reviewer Agent：检查质量
```

## 验收标准

给定任务：

```text
请分析这些资料并生成一份研究摘要。
```

系统能完成：

```text
任务拆解
资料检索
初稿生成
质量检查
最终输出
```

---

# Chapter 09：LangSmith Tracing 与 Debug

## 目标

让读者学会观察 Agent 每一步发生了什么。

## 内容

1. LangSmith 是什么。
2. 配置环境变量。
3. 开启 tracing。
4. 查看 run。
5. 查看 prompt、输入、输出、tool call。
6. 对比不同版本运行结果。
7. Debug 常见错误。
8. 记录 cost、latency、token usage。

## 产出文件

```text
examples/07_langsmith_tracing.py
docs/langsmith_tracing_guide.md
```

## 环境变量

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=agent-from-zero
```

## 验收标准

在 LangSmith UI 中可以看到：

```text
chain run
graph run
tool call
retriever call
LLM input/output
```

---

# Chapter 10：LangSmith Evaluation

## 目标

建立 Agent 的测试集和评估流程。

## 内容

1. 为什么 Agent 需要 evaluation。
2. Dataset。
3. Target function。
4. Evaluator。
5. LLM-as-judge。
6. Rule-based evaluator。
7. RAG 评估。
8. 回归测试。
9. 版本对比。

## 产出文件

```text
src/agent_from_zero/evaluation/datasets.py
src/agent_from_zero/evaluation/evaluators.py
src/agent_from_zero/evaluation/run_eval.py
examples/08_evaluation.py
```

## 示例评估维度

```text
答案是否相关
答案是否引用来源
答案是否幻觉
JSON 格式是否正确
工具调用是否正确
是否遵守系统指令
```

## 验收标准

运行：

```bash
python src/agent_from_zero/evaluation/run_eval.py
```

能够得到一组评估结果，并在 LangSmith 中查看实验对比。

---

# Chapter 11：FastAPI 与 Streamlit 演示

## 目标

把 Agent 从脚本升级成可交互应用。

## 内容

1. FastAPI 基础。
2. 定义 chat endpoint。
3. 定义 upload endpoint。
4. 定义 graph invoke endpoint。
5. Streamlit 前端。
6. 文件上传。
7. 流式输出。
8. 错误提示。

## 产出文件

```text
src/agent_from_zero/api/main.py
src/agent_from_zero/api/routers.py
src/agent_from_zero/api/schemas.py
src/agent_from_zero/ui/streamlit_app.py
scripts/run_api.py
scripts/run_ui.py
```

## API 示例

```text
POST /chat
POST /rag/query
POST /graph/invoke
POST /documents/upload
GET /health
```

## 验收标准

用户可以通过网页：

```text
上传文档
提问
查看回答
查看引用来源
触发 Agent 工作流
进行人工审核
```

---

# Chapter 12：生产化与最佳实践

## 目标

让读者理解 Demo 到产品之间的差距。

## 内容

1. API Key 安全。
2. Prompt 版本管理。
3. 日志系统。
4. 异常处理。
5. 并发与队列。
6. 缓存。
7. 权限控制。
8. 数据隔离。
9. 成本控制。
10. 模型切换。
11. 评估集持续更新。
12. Docker 部署。
13. CI/CD。
14. 生产监控。

## 产出文件

```text
Dockerfile
docker-compose.yml
docs/production_checklist.md
```

## 验收标准

项目可以通过 Docker Compose 启动：

```bash
docker compose up
```

并访问：

```text
FastAPI docs
Streamlit UI
```

---

## 6. Codex 生成教程时的统一要求

后续让 Codex 生成每一章时，应遵守以下规则。

### 6.1 每章文档格式

每章 Markdown 应包含：

```text
# Chapter XX：标题

## 本章目标
## 本章最终效果
## 前置知识
## 核心概念
## 项目文件
## 代码实现
## 运行方式
## 常见错误
## 扩展练习
## 本章小结
```

### 6.2 代码要求

所有代码必须满足：

```text
可直接运行
路径与仓库结构一致
包含必要注释
不硬编码 API Key
使用 .env 读取配置
有基础异常处理
函数职责清晰
命名清楚
```

### 6.3 教程风格

语言风格：

```text
面向初学者
先解释为什么，再解释怎么做
少讲抽象概念，多给可运行代码
每章都要有最小可运行示例
每章都要有验收标准
```

### 6.4 每章代码生成原则

每章新增代码必须：

```text
不破坏前面章节代码
可以独立运行对应 examples
尽量复用 src 中已有模块
避免把所有逻辑写在一个脚本里
保持教程渐进式演化
```

---

## 7. 关键概念解释要求

后续教程中必须解释清楚以下概念。

### 7.1 LangChain

需要解释：

```text
LangChain 解决什么问题
为什么不用纯 API
ChatModel 是什么
PromptTemplate 是什么
OutputParser 是什么
Runnable 是什么
Tool 是什么
Retriever 是什么
Agent 是什么
```

### 7.2 LangGraph

需要解释：

```text
为什么复杂 Agent 不适合只用 chain
State 是什么
Node 是什么
Edge 是什么
Conditional Edge 是什么
Checkpoint 是什么
Interrupt 是什么
Human-in-the-loop 是什么
Thread 是什么
Subgraph 是什么
```

### 7.3 LangSmith

需要解释：

```text
为什么 Agent 必须可观测
Tracing 是什么
Run 是什么
Project 是什么
Dataset 是什么
Evaluation 是什么
Experiment 是什么
如何定位 Agent 失败原因
如何比较两个 prompt 或两个模型
```

---

## 8. 示例业务任务设计

教程中建议贯穿同一个主线案例。

### 8.1 主线案例：研究资料分析 Agent

用户需求：

```text
我上传了一批研究资料，请帮我总结主要内容、提取关键信息，并生成一份结构化报告。
```

Agent 工作流：

```text
接收用户问题
↓
判断是否需要读取文档
↓
如果需要，执行 RAG 检索
↓
如果需要计算或写文件，调用工具
↓
生成初版回答
↓
质量检查
↓
必要时重试
↓
人工审核
↓
输出最终报告
```

### 8.2 可扩展到的实际场景

```text
论文阅读助手
审稿意见回复助手
专利分析助手
矿山资料分析助手
客户报告生成助手
代码知识库问答助手
企业内部知识库助手
```

---

## 9. 教程中的最小 Agent 演化路径

建议全教程按以下演化路线写。

### Version 0：普通 LLM 调用

```text
用户输入 → 模型回答
```

### Version 1：结构化输出

```text
用户输入 → Prompt → 模型 → JSON 输出
```

### Version 2：工具调用

```text
用户输入 → 模型判断 → 调用工具 → 汇总回答
```

### Version 3：RAG

```text
用户问题 → 检索文档 → 拼接上下文 → 模型回答
```

### Version 4：LangGraph 单流程

```text
输入 → 意图判断 → 分支执行 → 输出
```

### Version 5：LangGraph 可恢复流程

```text
输入 → 多节点执行 → checkpoint → 可恢复 → 输出
```

### Version 6：Human-in-the-loop

```text
生成初稿 → 暂停 → 人工审核 → 继续 → 最终输出
```

### Version 7：Multi-agent

```text
Planner → Researcher → Tool Executor → Writer → Reviewer
```

### Version 8：Production Agent

```text
API / UI / tracing / evaluation / deployment
```

---

## 10. 每章 Codex Prompt 模板

后续可用以下 Prompt 让 Codex 生成每章教程。

```text
请根据 design.md 中的 Chapter XX 设计，生成该章节的完整教程。

要求：
1. 输出 Markdown 教程文件。
2. 包含本章目标、概念解释、完整代码、运行方式、常见错误、练习。
3. 所有代码路径必须符合 design.md 的仓库结构。
4. 代码必须可以运行。
5. 不要跳过环境变量配置。
6. 不要硬编码 API Key。
7. 本章新增代码不能破坏之前章节。
8. 最后给出验收标准。
```

---

## 11. 单章代码生成 Prompt 模板

```text
请根据 design.md 和 Chapter XX 的要求，只生成本章涉及的代码文件。

要求：
1. 按文件路径分别输出。
2. 每个文件使用独立代码块。
3. 代码必须完整，不要省略。
4. 保留必要注释。
5. 使用 .env 管理配置。
6. 提供运行命令。
7. 提供简单测试方式。
```

---

## 12. 项目 README 生成 Prompt 模板

```text
请根据 design.md 生成项目 README.md。

README 需要包含：
1. 项目简介
2. 技术栈
3. 功能列表
4. 项目结构
5. 环境安装
6. .env 配置
7. 快速开始
8. 每章教程入口
9. 常见问题
10. 后续扩展路线
```

---

## 13. 测试设计

### 13.1 单元测试

需要覆盖：

```text
配置加载
LLM factory
tool function
RAG loader
retriever
graph node
output parser
```

### 13.2 集成测试

需要覆盖：

```text
完整 chat chain
完整 RAG chain
完整 LangGraph workflow
human-in-the-loop resume
LangSmith tracing 是否开启
```

### 13.3 示例测试命令

```bash
pytest tests/
```

---

## 14. 安全与边界设计

教程需要提醒读者：

```text
不要硬编码 API Key
不要让 Agent 任意执行系统命令
文件写入必须限制目录
数据库写操作需要人工确认
涉及删除、发送邮件、执行 SQL 等高风险操作必须 human-in-the-loop
日志中不要保存敏感信息
上传文件需要类型和大小限制
```

---

## 15. 常见问题章节设计

后续教程应覆盖以下问题：

```text
为什么环境变量没有生效？
为什么 LangSmith 没有记录 trace？
为什么模型没有调用工具？
为什么 RAG 回答胡编？
为什么检索不到相关文档？
为什么 LangGraph 状态没有更新？
为什么 interrupt 后无法恢复？
为什么输出 JSON 解析失败？
为什么多 Agent 陷入循环？
为什么成本突然变高？
```

---

## 16. 进阶扩展路线

完成基础教程后，可以继续扩展：

```text
接入数据库
接入企业私有知识库
接入网页搜索
接入文件上传
接入 OCR
接入图表生成
接入代码执行沙箱
接入地球物理正演/反演程序
接入报告自动生成
接入权限系统
接入多租户数据隔离
```

---

## 17. 推荐最终演示 Demo

最终 Demo 名称：

```text
Research Agent Studio
```

核心功能：

```text
上传文档
建立知识库
提问
调用工具
生成报告
人工审核
查看 LangSmith trace
运行 evaluation
```

演示流程：

```text
1. 启动 API。
2. 启动 Streamlit。
3. 上传示例文档。
4. 执行文档入库。
5. 提问。
6. 查看 RAG 引用。
7. 触发 LangGraph 工作流。
8. 在人工审核节点暂停。
9. 输入 approve。
10. 生成最终报告。
11. 打开 LangSmith 查看 trace。
12. 运行 evaluation 对比结果。
```

---

## 18. 版本规划

### v0.1

```text
环境搭建
普通 LLM 调用
LangChain 基础
```

### v0.2

```text
工具调用
结构化输出
基础 RAG
```

### v0.3

```text
LangGraph 基础流程
条件分支
状态管理
```

### v0.4

```text
checkpoint
human-in-the-loop
多 Agent
```

### v0.5

```text
LangSmith tracing
LangSmith evaluation
```

### v1.0

```text
FastAPI
Streamlit
Docker
完整教程
```

---

## 19. 开发顺序建议

推荐按以下顺序开发：

```text
1. 初始化项目
2. 配置管理
3. LLM factory
4. 简单 chain
5. 结构化输出
6. tools
7. RAG
8. basic graph
9. conditional graph
10. human review graph
11. LangSmith tracing
12. evaluation
13. FastAPI
14. Streamlit
15. Docker
16. README 和完整教程
```

---

## 20. 最终验收标准

整个教程完成后，需要满足：

```text
新手可以从零跟着跑通项目
每章都有独立运行示例
项目结构清晰
代码可维护
Agent 具备 RAG 和工具调用能力
LangGraph 工作流可运行
支持人工审核
支持 checkpoint
LangSmith 可看到完整 trace
LangSmith 可运行 evaluation
FastAPI 可调用
Streamlit 可交互
Docker 可启动
```

---

## 21. 官方资料参考

建议后续写教程时优先参考官方文档：

```text
LangChain Docs:
https://docs.langchain.com/

LangGraph Docs:
https://docs.langchain.com/oss/python/langgraph/overview

LangGraph Durable Execution:
https://docs.langchain.com/oss/python/langgraph/durable-execution

LangGraph Persistence:
https://docs.langchain.com/oss/python/langgraph/persistence

LangChain Human-in-the-loop:
https://docs.langchain.com/oss/python/langchain/human-in-the-loop

LangSmith Docs:
https://docs.langchain.com/langsmith/home

LangSmith Evaluation:
https://docs.langchain.com/langsmith/evaluation

LangSmith Evaluation Quickstart:
https://docs.langchain.com/langsmith/evaluation-quickstart

LangSmith Deployment:
https://docs.langchain.com/langsmith/deployment
```

---

## 22. 给 Codex 的总指令

```text
你是一个高级 Python + LangChain + LangGraph + LangSmith 教程工程师。

请严格根据 design.md 生成教程和代码。

核心原则：
1. 从零开始，逐步递进。
2. 每章都必须有可运行代码。
3. 不要一次性堆复杂概念。
4. 先让读者跑通，再解释原理。
5. 所有文件路径必须符合 design.md。
6. 所有配置必须通过 .env 管理。
7. 所有代码必须适合初学者阅读。
8. 每章都要给出运行命令和验收标准。
9. 需要说明常见错误和排查方法。
10. 最终项目必须具备工程化雏形。
```
