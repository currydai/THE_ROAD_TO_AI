from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


class GraphRequest(BaseModel):
    question: str


class UploadResponse(BaseModel):
    filename: str
    saved_path: str
