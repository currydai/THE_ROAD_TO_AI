# Chapter 03：LangChain 基础

## 本章目标

掌握 ChatModel、PromptTemplate、Runnable、LCEL 和 Pydantic 结构化输出。

## 本章最终效果

```bash
python examples/02_structured_output.py
```

输出 `ResearchSummary` JSON。

## 前置知识

已完成前两章，并能理解 Pydantic schema。

## 核心概念

LangChain 把模型、Prompt、解析器和工具统一成 Runnable。`prompt | model` 就是一个最小 LCEL chain。结构化输出可以通过模型的 `with_structured_output()` 获得更稳定的 schema 约束。

## 项目文件

- `src/agent_from_zero/chains/simple_chat.py`
- `src/agent_from_zero/chains/structured_output.py`
- `src/agent_from_zero/models/schemas.py`
- `examples/02_structured_output.py`

## 代码实现

`simple_chat.py` 构建最小聊天链，`structured_output.py` 把资料摘要约束成 `ResearchSummary`。

## 运行方式

```bash
python examples/02_structured_output.py
```

## 常见错误

- fake model 没有原生结构化输出：代码会回退到 JSON 解析和示例结果。
- schema 太复杂：先保持字段少而清楚，再逐步扩展。

## 扩展练习

增加一个 `confidence: float` 字段，并要求模型给出 0 到 1 的置信度。

## 本章小结

LangChain 的价值不是让简单调用更短，而是让 Prompt、模型、工具、检索和工作流能以统一接口组合。
