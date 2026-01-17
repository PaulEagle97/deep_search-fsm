from typing import List
from pydantic import BaseModel, HttpUrl


class ScrapedWebPage(BaseModel):
    url: HttpUrl
    title: str
    description: str
    content: str
    used_tokens: int


class JinaReaderSearchResult(BaseModel):
    query: str
    success: bool
    scraped_pages: List[ScrapedWebPage] = []
    total_used_tokens: int = 0
