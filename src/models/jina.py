from typing import List, Optional
from pydantic import BaseModel, HttpUrl

from .llm import PageEvaluation


class ScrapedWebPage(BaseModel):
    url: HttpUrl
    title: str
    description: str
    content: str
    used_tokens: int
    llm_eval: Optional[PageEvaluation] = None


class JinaReaderSearchResult(BaseModel):
    query: str
    success: bool
    scraped_pages: List[ScrapedWebPage] = []
    total_used_tokens: int = 0
