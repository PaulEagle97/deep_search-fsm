from typing import List
from pydantic import BaseModel, HttpUrl


class ScrapedWebPage(BaseModel):
    url: HttpUrl
    title: str
    description: str
    content: str
    jina_tokens: int
    openai_tokens: int


class JinaReaderSearchResult(BaseModel):
    query: str
    success: bool
    scraped_pages: List[ScrapedWebPage] = []
    total_jina_tokens: int = 0
