# Chapter 05：RAG 文档问答

## 本章目标

构建一个本地知识库问答系统。

## 本章最终效果

```bash
python examples/04_rag_qa.py
```

系统会读取示例资料、建立 FAISS 索引、回答问题并输出引用来源。

## 前置知识

了解 embedding、向量相似度和文档切分的基本概念。

## 核心概念

RAG 的流程是加载文档、切分 chunks、生成 embedding、写入向量库、检索相关 chunks、把上下文交给模型回答。引用来源能帮助用户判断答案依据。

## 项目文件

- `src/agent_from_zero/rag/loaders.py`
- `src/agent_from_zero/rag/splitters.py`
- `src/agent_from_zero/rag/vectorstore.py`
- `src/agent_from_zero/rag/retriever.py`
- `src/agent_from_zero/rag/qa_chain.py`
- `scripts/ingest_docs.py`
- `examples/04_rag_qa.py`

## 代码实现

当前版本支持 `.txt`、`.md`、`.csv`。没有资料时，`ensure_demo_index()` 会生成一个最小 demo 文档，保证示例可运行。

## 运行方式

```bash
python scripts/ingest_docs.py
python examples/04_rag_qa.py
```

如果 `data/raw` 为空，可直接运行示例，它会自动生成 demo 文档。

## 常见错误

- 检索不到：检查文件是否在 `data/raw`，后缀是否支持。
- 回答胡编：降低 chunk 数量、优化 prompt、要求引用来源。
- 向量库加载失败：删除旧的 `data/vectorstore` 后重新入库。

## 扩展练习

把 `k` 从 4 改成 2 或 8，对比回答质量和引用数量。

## 本章小结

RAG 是让 Agent 使用外部知识的基础，但检索质量、chunk 策略和引用机制同样重要。
