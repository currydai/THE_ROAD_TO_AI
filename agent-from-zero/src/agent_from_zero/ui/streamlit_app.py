import requests
import streamlit as st

API_BASE = st.sidebar.text_input("API Base", value="http://127.0.0.1:8000")

st.set_page_config(page_title="Research Agent Studio", layout="wide")
st.title("Research Agent Studio")

tab_chat, tab_rag, tab_graph = st.tabs(["Chat", "RAG", "Graph"])

with tab_chat:
    question = st.text_area("问题", key="chat_q", value="你好，请介绍 Agent。")
    if st.button("发送", key="chat_btn"):
        resp = requests.post(f"{API_BASE}/chat", json={"question": question}, timeout=60)
        st.markdown(resp.json()["answer"])

with tab_rag:
    file = st.file_uploader("上传资料", type=["txt", "md", "csv"])
    if file and st.button("上传"):
        resp = requests.post(
            f"{API_BASE}/documents/upload",
            files={"file": (file.name, file.getvalue())},
            timeout=60,
        )
        st.json(resp.json())
    rag_q = st.text_input("资料问题", value="这批文档主要讨论了什么？")
    if st.button("RAG 提问"):
        resp = requests.post(f"{API_BASE}/rag/query", json={"question": rag_q}, timeout=60)
        st.json(resp.json())

with tab_graph:
    graph_q = st.text_area("工作流输入", value="请根据资料生成一份研究摘要。")
    if st.button("运行 Graph"):
        resp = requests.post(f"{API_BASE}/graph/invoke", json={"question": graph_q}, timeout=60)
        st.markdown(resp.json()["answer"])
