from pathlib import Path

from langchain_community.vectorstores import FAISS

from agent_from_zero.config.settings import get_settings
from agent_from_zero.models.llm_factory import get_embedding_model


def build_faiss_index(documents, persist_dir: str | Path | None = None) -> FAISS:
    settings = get_settings()
    target = Path(persist_dir or settings.vectorstore_dir)
    target.mkdir(parents=True, exist_ok=True)
    store = FAISS.from_documents(documents, get_embedding_model())
    store.save_local(str(target))
    return store


def load_faiss_index(persist_dir: str | Path | None = None) -> FAISS:
    settings = get_settings()
    target = Path(persist_dir or settings.vectorstore_dir)
    return FAISS.load_local(
        str(target),
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )
