from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    intent: Literal["chat", "rag", "tool"]
    answer: str
    draft: str
    final_report: str
    retry_count: int
    approved: bool
