from typing import List, Optional
from pydantic import BaseModel, HttpUrl


class ScrapedWebPage(BaseModel):
    url: HttpUrl
    title: str
    description: str
    content: str
    jina_tokens: int
    content_tokens: Optional[int] = None


class JinaReaderSearchResult(BaseModel):
    query: str
    success: bool
    scraped_pages: List[ScrapedWebPage] = []
    total_jina_tokens: Optional[int] = None
