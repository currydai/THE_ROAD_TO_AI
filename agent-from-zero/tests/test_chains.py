from agent_from_zero.chains.simple_chat import chat
from agent_from_zero.chains.structured_output import summarize_research


def test_simple_chat() -> None:
    assert "Hello, Agent" in chat("Hello, Agent!")


def test_structured_output() -> None:
    result = summarize_research("Agent 包含模型、工具、RAG 和工作流。")
    assert result.summary
