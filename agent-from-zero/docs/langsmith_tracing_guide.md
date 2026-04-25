# LangSmith Tracing Guide

1. 在 `.env` 中设置：

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的_langsmith_api_key
LANGSMITH_PROJECT=agent-from-zero
```

2. 运行任意示例：

```bash
python examples/05_basic_langgraph.py
```

3. 打开 LangSmith 项目，检查 run 中的 prompt、input、output、tool call、retriever call 和 latency。

排查顺序：

- 当前 shell 是否加载了 `.env`
- project 名称是否正确
- 是否使用了 LangChain/LangGraph Runnable
- API Key 是否属于当前 workspace
