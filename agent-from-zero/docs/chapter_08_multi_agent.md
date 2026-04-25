# Chapter 08：多 Agent 工作流

## 本章目标

用 LangGraph 构建多个角色协作的 Agent 系统。

## 本章最终效果

```bash
python examples/07_multi_agent.py
```

输出一份包含计划、资料结论和质量检查的研究摘要。

## 前置知识

已掌握基础 LangGraph。

## 核心概念

多 Agent 不是越多越好。只有当任务天然包含不同职责时，才拆成 Planner、Researcher、Writer、Reviewer。每个角色都应该有清晰输入和输出，避免职责混乱和无限循环。

## 项目文件

- `src/agent_from_zero/graphs/multi_agent_graph.py`
- `examples/07_multi_agent.py`

## 代码实现

当前示例使用顺序图：Planner 制定计划，Researcher 做 RAG，Writer 写报告，Reviewer 做质量检查。

## 运行方式

```bash
python examples/07_multi_agent.py
```

## 常见错误

- 角色太多：先从 2 到 3 个角色开始。
- 无限循环：设置最大轮数或明确 END 条件。
- 状态污染：每个节点只更新自己负责的字段。

## 扩展练习

把 Reviewer 改成条件节点，如果报告太短就返回 Writer 重写。

## 本章小结

多 Agent 的价值在职责分解，不在角色数量。流程边界越清晰，系统越容易调试。
