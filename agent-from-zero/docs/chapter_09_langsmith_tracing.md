# Chapter 09：LangSmith Tracing 与 Debug

## 本章目标

学会观察 Agent 每一步发生了什么。

## 本章最终效果

开启 LangSmith 后运行：

```bash
python examples/07_langsmith_tracing.py
```

可以在 LangSmith 项目中看到 run。

## 前置知识

已跑通 chain、tool、RAG 和 graph。

## 核心概念

Agent 失败时，肉眼只看最终答案不够。Tracing 可以记录 prompt、输入、输出、tool call、retriever call、latency 和 token usage。LangSmith 的 Project 用来组织 run，后续 evaluation 会基于这些运行记录做对比。

## 项目文件

- `examples/07_langsmith_tracing.py`
- `docs/langsmith_tracing_guide.md`

## 代码实现

本项目通过环境变量启用 tracing：

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的_key
LANGSMITH_PROJECT=agent-from-zero
```

## 运行方式

```bash
python examples/07_langsmith_tracing.py
python examples/05_basic_langgraph.py
```

## 常见错误

- UI 中没有 run：确认环境变量在当前 shell 生效。
- project 不对：检查 `LANGSMITH_PROJECT`。
- 只看到部分 run：确认调用链经过 LangChain/LangGraph Runnable。

## 扩展练习

分别运行 fake model 和真实模型，对比 latency 与输出质量。

## 本章小结

可观测性不是上线后才补的功能。Agent 从开发第一天就应该能追踪。
