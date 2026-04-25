from langgraph.types import Command

from agent_from_zero.graphs.human_review_graph import build_human_review_graph


if __name__ == "__main__":
    graph = build_human_review_graph()
    config = {"configurable": {"thread_id": "demo-thread-1"}}
    first = graph.invoke({"question": "请生成一份 Agent 研究资料摘要。"}, config=config)
    print("已暂停，等待人工审核：")
    print(first["__interrupt__"])

    final = graph.invoke(Command(resume="approve"), config=config)
    print("\n最终报告：")
    print(final["final_report"])
