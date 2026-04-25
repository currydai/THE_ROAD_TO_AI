from langgraph.graph import END, START, StateGraph

from agent_from_zero.graphs.state import AgentState
from agent_from_zero.rag.qa_chain import answer_question


def planner(state: AgentState) -> AgentState:
    return {"draft": f"计划：检索资料，提炼要点，生成报告。任务：{state['question']}"}


def researcher(state: AgentState) -> AgentState:
    rag = answer_question(state["question"])
    return {"answer": rag.answer}


def writer(state: AgentState) -> AgentState:
    return {
        "final_report": (
            "# 研究摘要\n\n"
            f"## 任务\n{state['question']}\n\n"
            f"## 资料结论\n{state.get('answer', '')}\n\n"
            "## 建议\n继续补充高质量资料，并用评估集验证回答稳定性。"
        )
    }


def reviewer(state: AgentState) -> AgentState:
    report = state.get("final_report", "")
    return {"final_report": report + "\n\n## 质量检查\n已包含任务、资料结论和建议。"}


def build_multi_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)
    builder.add_node("reviewer", reviewer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_edge("reviewer", END)
    return builder.compile()
