from typing import List
from pydantic import BaseModel
from haystack.dataclasses import ChatMessage


class ApplicationState(BaseModel):
    counter: int = 0
    chat_history: List[ChatMessage] = []
    should_continue: bool = True
