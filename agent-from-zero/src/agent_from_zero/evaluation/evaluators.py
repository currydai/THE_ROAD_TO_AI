from agent_from_zero.models.schemas import EvaluationResult


def keyword_relevance(answer: str, expected_keywords: list[str]) -> EvaluationResult:
    if not expected_keywords:
        return EvaluationResult(name="keyword_relevance", score=1.0, comment="无关键词要求")
    hits = [word for word in expected_keywords if word in answer]
    score = len(hits) / len(expected_keywords)
    return EvaluationResult(
        name="keyword_relevance",
        score=score,
        comment=f"命中关键词：{', '.join(hits) if hits else '无'}",
    )


def has_citation(answer: str) -> EvaluationResult:
    ok = "引用" in answer or "source" in answer or "chunk" in answer
    return EvaluationResult(name="has_citation", score=1.0 if ok else 0.0, comment="检查引用来源")
