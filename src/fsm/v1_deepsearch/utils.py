import json
import logging
from pathlib import Path
from typing import Callable, List

from haystack.dataclasses import ChatMessage

from ...models import JinaReaderSearchResult, ScrapedWebPage, SearchReasoningNextQuery, SearchReasoningFollowUps
from ...tools import jina_search

from .prompt import (
    get_page_eval_sys_prompt,
    get_page_relevance_sys_prompt,
    get_page_depth_sys_prompt,
    get_page_eval_user_prompt_template,
    get_page_relevance_user_prompt_template,
    get_page_depth_user_prompt_template,
    get_iterative_searcher_next_query_sys_prompt,
    get_final_report_sys_prompt,
    get_final_report_user_prompt_template,
    get_iterative_searcher_user_prompt_template,
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


def build_iterative_searcher_msgs() -> List[ChatMessage]:
    sys_message = ChatMessage.from_system(get_iterative_searcher_next_query_sys_prompt())
    user_message = ChatMessage.from_user(get_iterative_searcher_user_prompt_template())

    return [sys_message, user_message]


def format_pages_for_report(pages: List[ScrapedWebPage]) -> str:
    if not pages:
        return "No sources available."
    
    formatted = []
    for idx, page in enumerate(pages, start=1):
        page_str = f"[{idx}] Title: {page.title}\n"
        page_str += f"[{idx}] URL: {page.url}\n"
        page_str += f"\n{page.content}"
        formatted.append(page_str)
    
    return "\n---\n".join(formatted)


def build_report_generator_msgs() -> List[ChatMessage]:
    sys_message = ChatMessage.from_system(get_final_report_sys_prompt())
    user_message = ChatMessage.from_user(get_final_report_user_prompt_template())

    return [sys_message, user_message]


def format_llm_reasoning_next_query(llm_reasoning: SearchReasoningNextQuery) -> str:
    return f"""**Evaluation:** {llm_reasoning.search_result_evaluation}
**Next Query:** {llm_reasoning.next_search_query}"""


def format_llm_reasoning_follow_ups(llm_reasoning: SearchReasoningFollowUps) -> str:
    follow_ups_formatted = "\n".join(f"- {direction}" for direction in llm_reasoning.search_result_follow_ups)
    return f"""**Evaluation:** {llm_reasoning.search_result_evaluation}
**Follow-up Directions:**
{follow_ups_formatted}"""


def count_content_tokens(search_result: JinaReaderSearchResult, tokenizer: Callable[[str], int]):
    total = 0
    for page in search_result.scraped_pages:
        content_tokens = tokenizer(page.content)
        page.content_tokens = content_tokens
        total += content_tokens
    
    return total, search_result


def trim_content_tokens(search_result: JinaReaderSearchResult, tokenizer: Callable[[str], int], token_limit: int):
    total = 0
    pages = []
    for page in search_result.scraped_pages:
        tokens_available = token_limit - total
        if page.content_tokens > tokens_available:
            to_keep_ratio = tokens_available / page.content_tokens
            page.content = page.content[:int(len(page.content) * to_keep_ratio)]
            page.content_tokens = tokenizer(page.content)
            pages.append(page)
            total += page.content_tokens
            break
        else:
            pages.append(page)
            total += page.content_tokens

    search_result.scraped_pages = pages

    return total, search_result
