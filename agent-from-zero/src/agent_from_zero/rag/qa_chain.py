from pathlib import Path

from agent_from_zero.models.llm_factory import get_chat_model
from agent_from_zero.models.schemas import Citation, RagAnswer
from agent_from_zero.rag.loaders import load_documents
from agent_from_zero.rag.splitters import split_documents
from agent_from_zero.rag.vectorstore import build_faiss_index, load_faiss_index


def ensure_demo_index() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    demo = raw_dir / "demo_research.md"
    if not demo.exists():
        demo.write_text(
            "# Agent 研究资料\n\n"
            "Agent 系统通常由大模型、工具调用、记忆、检索增强生成和工作流编排组成。\n"
            "LangGraph 适合把复杂任务拆成状态机节点，LangSmith 适合观察和评估运行结果。\n",
            encoding="utf-8",
        )
    chunks = split_documents(load_documents(raw_dir))
    if chunks:
        build_faiss_index(chunks)


def answer_question(question: str, k: int = 4) -> RagAnswer:
    try:
        store = load_faiss_index()
    except Exception:
        ensure_demo_index()
        store = load_faiss_index()

    docs = store.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[source={doc.metadata.get('source')} chunk={doc.metadata.get('chunk_id')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )
    prompt = f"基于以下资料回答问题，并在答案中说明依据。\n\n资料：\n{context}\n\n问题：{question}"
    result = get_chat_model().invoke(prompt)
    citations = [
        Citation(
            source=str(doc.metadata.get("source", "")),
            chunk_id=int(doc.metadata.get("chunk_id", 0)),
            text=doc.page_content[:300],
        )
        for doc in docs
    ]
    return RagAnswer(answer=str(result.content), citations=citations)
