from langgraph.types import Command

from agent_from_zero.graphs.basic_graph import build_basic_graph
from agent_from_zero.graphs.human_review_graph import build_human_review_graph


def test_basic_graph_tool_branch() -> None:
    result = build_basic_graph().invoke({"question": "请计算 18 * 27"})
    assert "486" in result["answer"]


def test_human_review_graph_resume() -> None:
    graph = build_human_review_graph()
    config = {"configurable": {"thread_id": "test-thread"}}
    first = graph.invoke({"question": "写一份短报告"}, config=config)
    assert "__interrupt__" in first
    final = graph.invoke(Command(resume="approve"), config=config)
    assert final["approved"] is True
