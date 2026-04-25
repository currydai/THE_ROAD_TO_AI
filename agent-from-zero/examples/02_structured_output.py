import json

from agent_from_zero.chains.structured_output import summarize_research


if __name__ == "__main__":
    summary = summarize_research(
        "LangGraph 使用状态机组织复杂 Agent。LangSmith 用于 tracing、debug 和 evaluation。"
    )
    print(json.dumps(summary.model_dump(), indent=2, ensure_ascii=False))
