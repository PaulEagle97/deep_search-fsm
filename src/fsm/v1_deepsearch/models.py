from typing import List
from pydantic import BaseModel
from haystack.dataclasses import ChatMessage

from ...models import JinaReaderSearchResult, ScrapedWebPage


class ApplicationState(BaseModel):
    user_query: str = ""
    next_search_query: str = ""
    final_report: str = ""
    executed_queries: List[str] = []
    msg_history: List[ChatMessage] = []
    search_results: List[JinaReaderSearchResult] = []
    report_sources: List[ScrapedWebPage] = []
    sources_token_counter: int = 0
    search_counter: int = 0
    continue_search: bool = True
