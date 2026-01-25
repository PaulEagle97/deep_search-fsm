from typing import List
from pydantic import BaseModel
from haystack.dataclasses import ChatMessage

from ...models import JinaReaderSearchResult


class ApplicationState(BaseModel):
    user_query: str = ""
    next_search_query: str = ""
    search_counter: int = 0
    token_counter: int = 0
    msg_history: List[ChatMessage] = []
    search_results: List[JinaReaderSearchResult] = []
    continue_search: bool = True
