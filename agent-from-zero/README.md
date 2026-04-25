# agent-from-zero

从 0 开始搭建一个基于 LangChain + LangGraph + LangSmith 的资料分析 Agent。项目主线是 `Research Agent Studio`：支持基础对话、结构化输出、工具调用、RAG 文档问答、LangGraph 工作流、人机审核、评估、FastAPI 和 Streamlit 演示。

## 技术栈

- Python 3.11+
- LangChain / LangGraph / LangSmith
- langchain-openai / langchain-community
- FAISS
- FastAPI / Streamlit
- pytest

## 快速开始

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
python examples/01_simple_llm.py
```

默认 `.env.example` 中 `USE_FAKE_MODEL=true`，没有 API Key 也能跑通教学示例。接入真实模型时，设置：

```bash
OPENAI_API_KEY=你的_key
USE_FAKE_MODEL=false
OPENAI_MODEL=gpt-4.1-mini
```

## 常用命令

```bash
python examples/02_structured_output.py
python examples/03_tool_calling.py
python examples/04_rag_qa.py
python examples/05_basic_langgraph.py
python examples/06_human_in_loop.py
python -m agent_from_zero.evaluation.run_eval
pytest
```

启动 API 和 UI：

```bash
python scripts/run_api.py
python scripts/run_ui.py
```

访问：

- FastAPI docs: http://127.0.0.1:8000/docs
- Streamlit: http://localhost:8501

## 教程目录

- [Chapter 01：环境搭建](docs/chapter_01_environment.md)
- [Chapter 02：LLM 调用基础](docs/chapter_02_llm_basics.md)
- [Chapter 03：LangChain 基础](docs/chapter_03_langchain_basics.md)
- [Chapter 04：Tool Calling 基础](docs/chapter_04_tool_calling.md)
- [Chapter 05：RAG 文档问答](docs/chapter_05_rag_basics.md)
- [Chapter 06：LangGraph 基础](docs/chapter_06_langgraph_basics.md)
- [Chapter 07：LangGraph 进阶](docs/chapter_07_langgraph_advanced.md)
- [Chapter 08：多 Agent 工作流](docs/chapter_08_multi_agent.md)
- [Chapter 09：LangSmith Tracing](docs/chapter_09_langsmith_tracing.md)
- [Chapter 10：LangSmith Evaluation](docs/chapter_10_langsmith_evaluation.md)
- [Chapter 11：FastAPI 与 Streamlit](docs/chapter_11_api_and_ui.md)
- [Chapter 12：生产化与最佳实践](docs/chapter_12_production.md)

## 项目结构

```text
src/agent_from_zero/
  config/       配置和日志
  models/       模型工厂和数据 schema
  chains/       LangChain 示例链
  tools/        工具函数
  rag/          文档加载、切分、向量库、问答
  graphs/       LangGraph 工作流
  evaluation/   评估数据和评估器
  api/          FastAPI
  ui/           Streamlit
examples/      每章最小可运行示例
docs/          完整教程
tests/         基础测试
```

## LangSmith

在 `.env` 中开启：

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的_langsmith_key
LANGSMITH_PROJECT=agent-from-zero
```

开启后，chain、graph、tool、retriever 运行会出现在 LangSmith 项目中，便于 debug 和 evaluation 对比。

## 安全边界

本教程刻意限制工具能力：计算器只允许数学表达式，文件读取只允许 `data/` 下的文本文件，报告写入固定目录。生产环境中，删除文件、发送邮件、数据库写操作、执行 SQL 等动作必须加入 human-in-the-loop 审核。
