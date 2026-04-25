# Chapter 11：FastAPI 与 Streamlit 演示

## 本章目标

把 Agent 从脚本升级成可交互应用。

## 本章最终效果

启动 API 和 UI：

```bash
python scripts/run_api.py
python scripts/run_ui.py
```

可以上传文档、提问、运行 Graph。

## 前置知识

了解 HTTP API、JSON 请求和文件上传。

## 核心概念

脚本适合学习，API 适合集成，UI 适合演示。FastAPI 暴露稳定接口，Streamlit 提供快速交互页面。

## 项目文件

- `src/agent_from_zero/api/main.py`
- `src/agent_from_zero/api/routers.py`
- `src/agent_from_zero/api/schemas.py`
- `src/agent_from_zero/ui/streamlit_app.py`
- `scripts/run_api.py`
- `scripts/run_ui.py`

## 代码实现

API 提供：

- `GET /health`
- `POST /chat`
- `POST /rag/query`
- `POST /graph/invoke`
- `POST /documents/upload`

## 运行方式

```bash
python scripts/run_api.py
python scripts/run_ui.py
```

## 常见错误

- Streamlit 请求失败：确认 API 已在 `127.0.0.1:8000` 启动。
- 上传失败：只支持 `.txt/.md/.csv`，大小不超过 2MB。
- RAG 不更新：上传后重新运行 `python scripts/ingest_docs.py`。

## 扩展练习

在 UI 中增加一个按钮，上传后自动调用入库脚本或 API。

## 本章小结

API/UI 是演示闭环，但生产系统还需要鉴权、限流、异步任务和审计日志。
