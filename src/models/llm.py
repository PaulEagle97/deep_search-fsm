from pydantic import BaseModel, Field


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


class SearchReasoningNextQuery(BaseModel):
    """Structured reasoning about search results and next steps."""
    search_result_evaluation: str = Field(
        description="How the latest search results contribute to the task."
    )
    next_search_query: str = Field(
        description="What should be searched for next to diversify already-seen web sources."
    )


class SearchReasoningFollowUps(BaseModel):
    """Structured reasoning about search results and next steps."""
    search_result_evaluation: str = Field(
        description="How the latest search results contribute to the task."
    )
    search_result_follow_ups: list[str] = Field(
        description="Promising follow-up directions discovered in the latest search results."
    )
