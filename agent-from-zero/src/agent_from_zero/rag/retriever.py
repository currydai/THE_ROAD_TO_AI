from agent_from_zero.rag.vectorstore import load_faiss_index


def get_retriever(k: int = 4):
    return load_faiss_index().as_retriever(search_kwargs={"k": k})
