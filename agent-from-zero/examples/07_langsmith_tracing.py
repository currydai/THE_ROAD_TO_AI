from agent_from_zero.chains.simple_chat import chat


if __name__ == "__main__":
    print("如需在 LangSmith 查看 trace，请在 .env 设置 LANGSMITH_TRACING=true 和 API Key。")
    print(chat("请用一句话解释 LangSmith tracing。"))
