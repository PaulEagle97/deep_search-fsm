from pydantic import BaseModel


class PageRelevanceEvaluation(BaseModel):
    summary: str
    relevance_score: int


class PageDepthEvaluation(BaseModel):
    summary: str
    depth_score: int


class PageEvaluation(BaseModel):
    depth_summary: str
    depth_score: int
    relevance_summary: str
    relevance_score: int


class PageEvaluationSeparate(BaseModel):
    relevance_evaluation: PageRelevanceEvaluation
    depth_evaluation: PageDepthEvaluation
