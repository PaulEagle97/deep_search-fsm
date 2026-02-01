import json
import logging
from pathlib import Path
from typing import List

from haystack.dataclasses import ChatMessage, ToolCall

from src.models.jina import ScrapedWebPage
from src.models.llm import SearchReasoning

from ...models import JinaReaderSearchResult
from ...tools import jina_search

from .prompt import (
    get_page_eval_sys_prompt,
    get_page_relevance_sys_prompt,
    get_page_depth_sys_prompt,
    get_page_eval_user_prompt_template,
    get_page_relevance_user_prompt_template,
    get_page_depth_user_prompt_template,
)
logger = logging.getLogger(__name__)

# Cache file path - in the same directory as temp.py
CACHE_FILE = Path(__file__).parent / "jina_search_cache.json"


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_cached_or_fetch(query: str, num_pages: int) -> JinaReaderSearchResult:    
    cache = load_cache()
    cache_key = f"{query}|{num_pages}"
    
    if cache_key in cache:
        logger.info(f"Cache HIT for query: '{query[:50]}...'")
        search_result = JinaReaderSearchResult.model_validate(cache[cache_key])
        logger.info(f"Read {len(search_result.scraped_pages)} cached pages")
    
    else:
        logger.info(f"Cache MISS for query: '{query[:50]}...' - fetching from Jina API")
        search_result = jina_search(query, num_pages)
        
        logger.info(f"Number of burned Jina API tokens: {search_result.total_jina_tokens}")
        logger.info(f"Jina API returned {len(search_result.scraped_pages)} pages")

        # Save to cache (using model_dump with mode='json' for JSON-serializable output)
        cache[cache_key] = search_result.model_dump(mode='json')
        save_cache(cache)
    
    return search_result


def build_page_relevance_msgs() -> List[ChatMessage]:
    sys_message = ChatMessage.from_system(get_page_relevance_sys_prompt())
    user_message = ChatMessage.from_user(get_page_relevance_user_prompt_template())

    return [sys_message, user_message]


def build_page_depth_msgs() -> List[ChatMessage]:
    sys_message = ChatMessage.from_system(get_page_depth_sys_prompt())
    user_message = ChatMessage.from_user(get_page_depth_user_prompt_template())

    return [sys_message, user_message]


def build_page_evaluation_msgs() -> List[ChatMessage]:
    sys_message = ChatMessage.from_system(get_page_eval_sys_prompt())
    user_message = ChatMessage.from_user(get_page_eval_user_prompt_template())

    return [sys_message, user_message]


def page_is_good_enough(page: ScrapedWebPage):
    return page.llm_eval.depth_score > 2 and page.llm_eval.relevance_score > 2


def format_evaluated_pages_for_llm(pages: List[ScrapedWebPage]) -> str:
    """
    Convert a list of evaluated web pages to a formatted string for LLM consumption.
    Shows page metadata, evaluation scores, and summaries.
    """
    if not pages:
        return "No pages were found or evaluated."
    
    sections = []
    for idx, page in enumerate(pages, 1):
        section = f"[{idx}] {page.title}\n"
        section += f"    URL: {page.url}\n"
        
        if page.llm_eval:
            eval = page.llm_eval
            section += f"    Relevance: {eval.relevance_score}/5 - {eval.relevance_summary}\n"
            section += f"    Depth: {eval.depth_score}/5 - {eval.depth_summary}\n"
        else:
            section += "    (not evaluated)\n"
        
        sections.append(section)
    
    # Add summary stats
    evaluated = [p for p in pages if p.llm_eval]
    if evaluated:
        avg_rel = sum(p.llm_eval.relevance_score for p in evaluated) / len(evaluated)
        avg_depth = sum(p.llm_eval.depth_score for p in evaluated) / len(evaluated)
        header = f"Found {len(pages)} pages (avg relevance: {avg_rel:.1f}/5, avg depth: {avg_depth:.1f}/5)\n\n"
    else:
        header = f"Found {len(pages)} pages\n\n"
    
    return header + "\n".join(sections)


def build_tool_msg_from_evals(pages: List[ScrapedWebPage], tool_call: ToolCall) -> ChatMessage:
    """Build a tool result message from evaluated pages."""
    formatted = format_evaluated_pages_for_llm(pages)
    
    return ChatMessage.from_tool(
        tool_result=formatted,
        origin=tool_call,
    )


def format_llm_reasoning(llm_reasoning: SearchReasoning) -> str:
    return f"**Evaluation:** {llm_reasoning.search_result_evaluation}\n**Next Query:** {llm_reasoning.next_search_query}"
