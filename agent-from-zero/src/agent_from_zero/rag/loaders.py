from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".txt", ".md", ".csv"}


def load_documents(raw_dir: str | Path) -> list[Document]:
    base = Path(raw_dir)
    if not base.exists():
        return []

    documents: list[Document] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": str(path)}))
    return documents
