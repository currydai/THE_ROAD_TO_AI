from agent_from_zero.rag.qa_chain import answer_question, ensure_demo_index


if __name__ == "__main__":
    ensure_demo_index()
    result = answer_question("这批文档主要讨论了什么？")
    print(result.answer)
    print("\n引用来源：")
    for citation in result.citations:
        print(f"- {citation.source} chunk={citation.chunk_id}")
