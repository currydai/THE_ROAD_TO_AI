from agent_from_zero.rag.qa_chain import answer_question, ensure_demo_index


def test_rag_answer_has_citations() -> None:
    ensure_demo_index()
    result = answer_question("Agent 系统由哪些部分组成？")
    assert result.answer
    assert result.citations
