# Chapter 04：Tool Calling 基础

## 本章目标

让 Agent 能调用外部函数，例如计算、读文件和写报告。

## 本章最终效果

```bash
python examples/03_tool_calling.py
```

会计算 `18 * 27` 并写入 Markdown 报告。

## 前置知识

理解 Python 函数、异常处理和文件路径。

## 核心概念

Tool 是给模型或工作流调用的受控函数。工具描述要写清楚输入、输出和边界。高风险工具必须人工审核。

## 项目文件

- `src/agent_from_zero/tools/calculator.py`
- `src/agent_from_zero/tools/file_reader.py`
- `src/agent_from_zero/tools/report_writer.py`
- `examples/03_tool_calling.py`

## 代码实现

计算器使用 `ast` 解析表达式，只允许基础数学运算。文件读取限制在 `data/` 目录，报告只能写到 `data/processed/reports/`。

## 运行方式

```bash
python examples/03_tool_calling.py
```

## 常见错误

- 不要用 `eval()` 执行用户输入。
- 不要允许 Agent 任意读取磁盘。
- 写文件、删文件、发邮件这类动作要加入 human-in-the-loop。

## 扩展练习

给报告工具增加 `tags` 参数，并把 tags 写入 Markdown frontmatter。

## 本章小结

Tool calling 的重点不是“能调用函数”，而是让模型在受控边界内调用函数。
