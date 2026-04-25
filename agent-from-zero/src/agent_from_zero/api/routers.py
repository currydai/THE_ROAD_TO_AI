from pathlib import Path
import json

from fastapi import APIRouter, File, HTTPException, UploadFile

from agent_from_zero.api.schemas import ChatRequest, ChatResponse, GraphRequest, UploadResponse
from agent_from_zero.chains.simple_chat import chat
from agent_from_zero.config.settings import get_settings
from agent_from_zero.graphs.basic_graph import build_basic_graph
from agent_from_zero.rag.qa_chain import answer_question

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    return ChatResponse(answer=chat(request.question))


@router.post("/rag/query", response_model=ChatResponse)
def rag_endpoint(request: ChatRequest) -> ChatResponse:
    payload = answer_question(request.question).model_dump()
    return ChatResponse(answer=json.dumps(payload, ensure_ascii=False, indent=2))


@router.post("/graph/invoke", response_model=ChatResponse)
def graph_endpoint(request: GraphRequest) -> ChatResponse:
    result = build_basic_graph().invoke({"question": request.question})
    return ChatResponse(answer=result.get("answer", ""))


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".csv"}:
        raise HTTPException(status_code=400, detail="只支持 .txt/.md/.csv")
    content = await file.read()
    if len(content) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件不能超过 2MB")
    target = get_settings().data_dir / "raw" / Path(file.filename or "upload.txt").name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return UploadResponse(filename=target.name, saved_path=str(target))
