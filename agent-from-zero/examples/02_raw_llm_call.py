import json

from agent_from_zero.models.llm_factory import get_chat_model
from agent_from_zero.models.schemas import RawExtraction


if __name__ == "__main__":
    prompt = """
请从下面资料中提取 JSON，字段为 title、summary、keywords、action_items。
资料：Agent 项目需要先跑通模型调用，再逐步加入工具、RAG、工作流和评估。
"""
    raw = get_chat_model().invoke(prompt).content
    data = json.loads(raw)
    parsed = RawExtraction.model_validate(data)
    print(json.dumps(parsed.model_dump(), indent=2, ensure_ascii=False))
