# Chapter 02：不使用框架，理解 LLM 调用基础

## 本章目标

理解 messages、system/user/assistant、temperature、JSON 输出和错误处理。

## 本章最终效果

运行：

```bash
python examples/02_raw_llm_call.py
```

得到可以被 Pydantic 解析的 JSON。

## 前置知识

需要了解 JSON 和 Python 异常处理。

## 核心概念

大模型对话通常由消息列表组成。`system` 定义行为边界，`user` 是用户输入，`assistant` 是模型历史回复。结构化输出的关键是让模型只输出 JSON，并在代码侧用 Pydantic 校验。

## 项目文件

- `examples/02_raw_llm_call.py`
- `src/agent_from_zero/models/llm_factory.py`
- `src/agent_from_zero/models/schemas.py`

## 代码实现

示例先调用 `get_chat_model()`，再用 `json.loads()` 和 `RawExtraction.model_validate()` 验证输出。真实项目中不能相信模型天然稳定，必须加解析失败处理。

## 运行方式

```bash
python examples/02_raw_llm_call.py
```

## 常见错误

- `json.loads` 失败：模型输出了 Markdown 或解释文字。要强化 prompt，或改用 `with_structured_output`。
- 输出缺字段：Pydantic 会报校验错误，应把错误反馈给模型重试。

## 扩展练习

把 `RawExtraction` 加一个 `risk_level` 字段，并观察模型输出是否仍能通过校验。

## 本章小结

不要急着上框架。先理解原始输入输出格式，后面使用 LangChain 时才知道抽象层在帮你做什么。
