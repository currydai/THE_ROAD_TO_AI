from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent_from_zero.graphs.state import AgentState
from agent_from_zero.models.llm_factory import invoke_text


def generate_draft(state: AgentState) -> AgentState:
    draft = invoke_text("请为下面任务生成一版简短研究报告草稿：\n" + state["question"])
    return {"draft": draft, "retry_count": state.get("retry_count", 0)}


def quality_check(state: AgentState) -> Command:
    retry_count = state.get("retry_count", 0)
    draft = state.get("draft", "")
    if len(draft) < 20 and retry_count < 1:
        return Command(goto="revise", update={"retry_count": retry_count + 1})
    return Command(goto="human_review")


def revise(state: AgentState) -> AgentState:
    return {"draft": state.get("draft", "") + "\n\n补充：已根据质量检查增加背景、依据和建议。"}


def human_review(state: AgentState) -> AgentState:
    decision = interrupt(
        {
            "instruction": "请输入 approve 接受草稿，或直接输入修改后的最终文本。",
            "draft": state.get("draft", ""),
        }
    )
    if isinstance(decision, str) and decision.strip().lower() == "approve":
        return {"approved": True, "final_report": state.get("draft", "")}
    return {"approved": True, "final_report": str(decision)}


def build_human_review_graph():
    builder = StateGraph(AgentState)
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("quality_check", quality_check)
    builder.add_node("revise", revise)
    builder.add_node("human_review", human_review)
    builder.add_edge(START, "generate_draft")
    builder.add_edge("generate_draft", "quality_check")
    builder.add_edge("revise", "quality_check")
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=InMemorySaver())
