from agent_from_zero.rag.loaders import load_documents
from agent_from_zero.rag.splitters import split_documents
from agent_from_zero.rag.vectorstore import build_faiss_index


def main() -> None:
    docs = load_documents("data/raw")
    chunks = split_documents(docs)
    if not chunks:
        raise SystemExit("data/raw 中没有可入库的 .txt/.md/.csv 文件")
    build_faiss_index(chunks)
    print(f"已入库 {len(chunks)} 个 chunks")


if __name__ == "__main__":
    main()
