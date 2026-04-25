# Chapter 01：环境搭建

## 本章目标

完成项目初始化，理解 `.env` 配置，并成功运行第一次模型调用。

## 本章最终效果

运行：

```bash
python examples/01_simple_llm.py
```

可以输出 `Hello, Agent!` 或模型回答。

## 前置知识

需要会使用命令行、Python 虚拟环境和基础 Git。

## 核心概念

Agent 项目不要把 API Key 写进代码。配置应该集中在 `.env`，代码通过 `settings.py` 读取。为了让新手没有 API Key 也能学习，本项目默认开启 `USE_FAKE_MODEL=true`。

## 项目文件

- `.env.example`
- `pyproject.toml`
- `src/agent_from_zero/config/settings.py`
- `src/agent_from_zero/models/llm_factory.py`
- `examples/01_simple_llm.py`

## 代码实现

`settings.py` 负责读取环境变量；`llm_factory.py` 负责根据配置返回真实 `ChatOpenAI` 或本地 fake model。所有后续章节都复用这个入口。

## 运行方式

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
python examples/01_simple_llm.py
```

接入真实模型：

```bash
OPENAI_API_KEY=你的_key
USE_FAKE_MODEL=false
```

## 常见错误

- `ModuleNotFoundError: agent_from_zero`：确认已经执行 `uv pip install -e ".[dev]"`。
- 模型鉴权失败：检查 `.env` 中的 `OPENAI_API_KEY`。
- 没有 LangSmith trace：检查 `LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY`。

## 扩展练习

把 `OPENAI_MODEL` 改成你的供应商兼容模型，并通过 `OPENAI_BASE_URL` 指向兼容 OpenAI Chat Completions 的地址。

## 本章小结

本章把配置、模型工厂和第一个示例跑通。后续所有能力都建立在这个稳定入口上。
