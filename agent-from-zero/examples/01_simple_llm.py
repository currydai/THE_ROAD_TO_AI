from agent_from_zero.models.llm_factory import get_chat_model


if __name__ == "__main__":
    result = get_chat_model().invoke("请回复：Hello, Agent!")
    print(result.content)
