from langchain_core.prompts import ChatPromptTemplate

from agent_from_zero.models.llm_factory import get_chat_model


def build_simple_chat_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个耐心、严谨的研究资料分析助手。"),
            ("human", "{question}"),
        ]
    )
    return prompt | get_chat_model()


def chat(question: str) -> str:
    result = build_simple_chat_chain().invoke({"question": question})
    return str(result.content)
