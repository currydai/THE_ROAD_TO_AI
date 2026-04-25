import json

from pydantic import ValidationError

from agent_from_zero.models.llm_factory import get_chat_model
from agent_from_zero.models.schemas import ResearchSummary


def summarize_research(text: str) -> ResearchSummary:
    model = get_chat_model()
    if hasattr(model, "with_structured_output"):
        structured = model.with_structured_output(ResearchSummary)
        return structured.invoke(
            "请把下面资料整理成 ResearchSummary 结构。资料：\n" + text
        )

    prompt = f"""
请只输出 JSON，不要输出 Markdown。字段包括：
topic, summary, key_points, limitations, next_steps。

资料：
{text}
"""
    raw = model.invoke(prompt).content
    try:
        return ResearchSummary.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError):
        return ResearchSummary(
            topic="本地示例主题",
            summary=str(raw),
            key_points=["已生成摘要，但模型未严格返回目标 schema"],
            limitations=["当前使用 fake model 或模型输出格式不稳定"],
            next_steps=["设置 OPENAI_API_KEY 并关闭 USE_FAKE_MODEL 后重试"],
        )
