from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from agent_from_zero.config.settings import get_settings


def _content_from_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, BaseMessage):
                parts.append(str(item.content))
            elif isinstance(item, dict):
                parts.append(str(item.get("content", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        if "messages" in value:
            return _content_from_input(value["messages"])
        return "\n".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def create_fake_chat_model() -> RunnableLambda:
    """A deterministic local model used when no API key is available."""

    def invoke(value: Any) -> AIMessage:
        text = _content_from_input(value)
        if "18 * 27" in text or "18*27" in text:
            return AIMessage(content="18 * 27 = 486")
        if "JSON" in text.upper() or "json" in text:
            return AIMessage(
                content=(
                    '{"title":"本地示例","summary":"这是一个无需 API Key 的结构化示例",'
                    '"keywords":["agent","langchain"],"action_items":["继续学习下一章"]}'
                )
            )
        return AIMessage(content=f"Hello, Agent! 我收到了你的输入：{text[:120]}")

    return RunnableLambda(invoke)


def get_chat_model(temperature: float = 0.2):
    settings = get_settings()
    if settings.use_fake_model or not settings.has_openai_key:
        return create_fake_chat_model()

    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.openai_model,
        "temperature": temperature,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def get_embedding_model():
    settings = get_settings()
    if settings.use_fake_model or not settings.has_openai_key:
        from langchain_core.embeddings import Embeddings

        class LocalHashEmbeddings(Embeddings):
            def _embed(self, text: str) -> list[float]:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                values = [((digest[i % len(digest)] / 255.0) * 2) - 1 for i in range(384)]
                return values

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self._embed(text) for text in texts]

            def embed_query(self, text: str) -> list[float]:
                return self._embed(text)

        return LocalHashEmbeddings()

    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, Any] = {
        "model": settings.openai_embedding_model,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)


def invoke_text(prompt: str) -> str:
    message = get_chat_model().invoke(prompt)
    return str(getattr(message, "content", message))
