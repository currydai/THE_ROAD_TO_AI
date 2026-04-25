from agent_from_zero.graphs.multi_agent_graph import build_multi_agent_graph


if __name__ == "__main__":
    result = build_multi_agent_graph().invoke({"question": "请分析这些资料并生成一份研究摘要。"})
    print(result["final_report"])
