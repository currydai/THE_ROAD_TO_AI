from agent_from_zero.graphs.basic_graph import build_basic_graph


if __name__ == "__main__":
    graph = build_basic_graph()
    result = graph.invoke({"question": "请计算 18 * 27"})
    print(result["answer"])
