from agent_from_zero.graphs.basic_graph import build_basic_graph


if __name__ == "__main__":
    graph = build_basic_graph()
    for question in ["你好", "这批文档主要讨论了什么？", "请计算 18 * 27"]:
        result = graph.invoke({"question": question})
        print(f"\nQ: {question}\nA: {result['answer']}")
