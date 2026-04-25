# Chapter 10：LangSmith Evaluation

## 本章目标

建立 Agent 的测试集和评估流程。

## 本章最终效果

```bash
python examples/08_evaluation.py
```

输出关键词相关性和引用检查结果。

## 前置知识

理解 RAG 问答和 LangSmith tracing。

## 核心概念

Evaluation 用数据集和评估器衡量 Agent 行为。基础版本可以用 rule-based evaluator，例如关键词命中、是否包含引用。进阶版本可以使用 LLM-as-judge，对相关性、忠实度、格式和安全性打分。

## 项目文件

- `src/agent_from_zero/evaluation/datasets.py`
- `src/agent_from_zero/evaluation/evaluators.py`
- `src/agent_from_zero/evaluation/run_eval.py`
- `examples/08_evaluation.py`

## 代码实现

`load_demo_dataset()` 提供最小测试集，`keyword_relevance()` 和 `has_citation()` 提供基础评估器。

## 运行方式

```bash
python examples/08_evaluation.py
```

## 常见错误

- 分数偏低：fake model 回答固定，真实模型效果会不同。
- 评估集太少：至少覆盖正常问题、边界问题和反例。
- 只看平均分：还要看失败样本。

## 扩展练习

新增 `json_format_valid` evaluator，用来检查结构化输出是否可解析。

## 本章小结

评估让 Agent 从“看起来能用”变成“可以持续改进”。
