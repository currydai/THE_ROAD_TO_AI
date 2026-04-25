from pydantic import BaseModel, Field


class ResearchSummary(BaseModel):
    topic: str = Field(description="资料主题")
    summary: str = Field(description="简洁摘要")
    key_points: list[str] = Field(default_factory=list, description="关键要点")
    limitations: list[str] = Field(default_factory=list, description="局限性")
    next_steps: list[str] = Field(default_factory=list, description="下一步建议")


class RawExtraction(BaseModel):
    title: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    source: str
    chunk_id: int
    text: str


class RagAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    name: str
    score: float
    comment: str = ""
