from dataclasses import dataclass


@dataclass(frozen=True)
class EvalExample:
    question: str
    expected_keywords: list[str]


def load_demo_dataset() -> list[EvalExample]:
    return [
        EvalExample("Agent 系统由哪些部分组成？", ["模型", "工具", "检索"]),
        EvalExample("LangGraph 适合解决什么问题？", ["工作流", "状态"]),
        EvalExample("LangSmith 可以用来做什么？", ["观察", "评估"]),
    ]
