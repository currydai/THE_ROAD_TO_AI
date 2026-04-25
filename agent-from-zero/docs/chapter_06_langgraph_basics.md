# Chapter 06：LangGraph 基础

## 本章目标

用 LangGraph 把单步 chain 改造成多步骤工作流。

## 本章最终效果

```bash
python examples/05_basic_langgraph.py
```

普通问题走 chat，资料问题走 RAG，计算问题走 tool。

## 前置知识

已经理解 chain、tool 和 RAG。

## 核心概念

复杂 Agent 不适合只用单条 chain，因为真实任务有分支、循环、失败恢复和人工审核。LangGraph 用 `State` 表示共享状态，用 `Node` 处理状态，用 `Edge` 连接步骤，用 conditional edge 做分支。

## 项目文件

- `src/agent_from_zero/graphs/state.py`
- `src/agent_from_zero/graphs/basic_graph.py`
- `examples/05_basic_langgraph.py`

## 代码实现

`classify_intent` 根据问题判断意图，`route_by_intent` 把状态路由到 `chat`、`rag` 或 `tool` 节点。

## 运行方式

```bash
python examples/05_basic_langgraph.py
```

## 常见错误

- 状态没有更新：节点必须返回要更新的字段字典。
- 条件分支找不到节点：条件函数返回值必须在 mapping 中。
- 工具节点失败：先把工具函数单独跑通。

## 扩展练习

给 intent 增加 `report` 分支，调用报告写入工具。

## 本章小结

LangGraph 的核心是显式建模流程。越复杂的 Agent，越需要把状态、节点和分支写清楚。
