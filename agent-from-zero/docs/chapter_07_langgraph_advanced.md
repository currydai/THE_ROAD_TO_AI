# Chapter 07：LangGraph 进阶：循环、重试与人机协同

## 本章目标

构建可暂停、可恢复、可人工审核的 Agent 工作流。

## 本章最终效果

```bash
python examples/06_human_in_loop.py
```

Graph 会在审核节点暂停，然后用 `Command(resume="approve")` 恢复。

## 前置知识

理解 LangGraph 的 State、Node、Edge。

## 核心概念

`interrupt()` 会暂停图执行，并把 payload 返回给调用方。恢复时必须使用相同的 `thread_id`，并通过 `Command(resume=...)` 传入人工决定。checkpoint 是恢复的基础，教程中用内存 checkpointer，生产环境应使用数据库 checkpointer。

## 项目文件

- `src/agent_from_zero/graphs/human_review_graph.py`
- `examples/06_human_in_loop.py`

## 代码实现

工作流包含 `generate_draft`、`quality_check`、`revise` 和 `human_review`。质量检查不合格时会重试一次；进入人工审核后暂停。

## 运行方式

```bash
python examples/06_human_in_loop.py
```

## 常见错误

- 恢复失败：确认恢复时使用同一个 `thread_id`。
- interrupt 被吞掉：不要把 `interrupt()` 包在宽泛 `try/except` 中。
- 重复执行：恢复时节点会从头运行，所以 interrupt 前的代码要保持幂等。

## 扩展练习

把 `approve`、`revise`、`reject` 三种决策都实现出来。

## 本章小结

human-in-the-loop 是安全 Agent 的基础能力。关键操作不要让模型直接执行。
