from langgraph.graph import END, START, StateGraph

from agent_from_zero.chains.simple_chat import chat
from agent_from_zero.graphs.state import AgentState
from agent_from_zero.rag.qa_chain import answer_question
from agent_from_zero.tools.calculator import calculator


def classify_intent(state: AgentState) -> AgentState:
    q = state["question"]
    if any(word in q for word in ["文档", "资料", "引用", "知识库"]):
        intent = "rag"
    elif any(token in q for token in ["计算", "*", "+", "-", "/", "乘以"]):
        intent = "tool"
    else:
        intent = "chat"
    return {"intent": intent}


def chat_node(state: AgentState) -> AgentState:
    return {"answer": chat(state["question"])}


def rag_node(state: AgentState) -> AgentState:
    result = answer_question(state["question"])
    sources = "\n".join(f"- {c.source} chunk={c.chunk_id}" for c in result.citations)
    return {"answer": f"{result.answer}\n\n引用来源：\n{sources}"}


def tool_node(state: AgentState) -> AgentState:
    text = state["question"].replace("请计算", "").replace("计算", "").strip(" 。")
    try:
        value = calculator.invoke({"expression": text})
        return {"answer": f"计算结果：{value}"}
    except Exception as exc:
        return {"answer": f"工具调用失败：{exc}"}


def route_by_intent(state: AgentState) -> str:
    return state["intent"]


def build_basic_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("chat", chat_node)
    graph.add_node("rag", rag_node)
    graph.add_node("tool", tool_node)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"chat": "chat", "rag": "rag", "tool": "tool"},
    )
    graph.add_edge("chat", END)
    graph.add_edge("rag", END)
    graph.add_edge("tool", END)
    return graph.compile()
