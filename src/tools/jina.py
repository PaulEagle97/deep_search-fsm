import logging
import requests
from urllib.parse import urlencode
from typing import List, Dict, Annotated, Optional

import pydantic

from rich.logging import RichHandler

from ..core import jina_config
from ..models import JinaReaderSearchResult, ScrapedWebPage

logger = logging.getLogger(__name__)


def jina_search(query: str, max_results: int) -> JinaReaderSearchResult:
    base_url = "https://s.jina.ai/"
    
    # Build query parameters
    params = {
        "q": query,
        "num": max_results
    }
    search_url = f"{base_url}?{urlencode(params)}"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {jina_config.envs.API_KEY}",
        "X-Retain-Images": "none",
        # to add "links" section with aggregated links per page
        # "X-With-Links-Summary": "true",
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=60)
        response.raise_for_status()
        
        search_results = response.json()

        pages = [
            ScrapedWebPage(
                url=page["url"],
                title=page["title"],
                description=page["description"],
                content=page["content"],
                used_tokens=page["usage"]["tokens"],
            ) for page in search_results["data"]
        ]

        return JinaReaderSearchResult(
            query=query,
            success=True,
            scraped_pages=pages,
            total_used_tokens=search_results["meta"]["usage"]["tokens"]
        )

    except requests.exceptions.RequestException:
        logger.exception("Jina API request failed")

    except pydantic.ValidationError:
        logger.exception("Jina API response parsing failed")

    return JinaReaderSearchResult(
        query=query,
        success=False,
    )


def jina_result_to_formatted_strs(search_result: JinaReaderSearchResult) -> List[str]:
    if not search_result.success:
        return ["Web search failed, try another time"]
    
    formatted_pages = []
    
    for idx, page in enumerate(search_result.scraped_pages, start=1):
        page_str = f"[{idx}] Title: {page.title}\n"
        page_str += f"[{idx}] URL Source: {page.url}\n"
        page_str += f"[{idx}] Description: {page.description}\n"
        page_str += "\n"
        page_str += page.content
        
        formatted_pages.append(page_str)
    
    return formatted_pages


def search_web_formatted_str_out(
    query: Annotated[str, "A query to be searched in the web"],
) -> List[Optional[str]]:
    """
    Performs a web search and returns scraped web page texts.
    """
    logger.info(f"Calling Jina API with query='{query}'")

    # SHORT CIRCUIT FOR TESTING
    # return ["Web Search failed due to the rate limits."]

    search_result = jina_search(query, jina_config.NUM_PAGES_PER_SEARCH)
    if search_result.success:
        logger.info(f"Jina API returned {len(search_result.scraped_pages)} pages")
    else:
        logger.warning(f"Failure to call Jina API")

    logger.info(f"Number of burned Jina API tokens: {search_result.total_used_tokens}")

    formatted_strs = jina_result_to_formatted_strs(search_result)

    return formatted_strs


def search_web_structured_out(
    query: Annotated[str, "A query to be searched in the web"],
) -> Dict:
    """
    Performs a web search and returns scraped web pages as dictionaries.
    """
    logger.info(f"Calling Jina API with query='{query}'")

    # SHORT CIRCUIT FOR TESTING
    # return ["Web Search failed due to the rate limits."]

    search_result = jina_search(query, jina_config.NUM_PAGES_PER_SEARCH)
    if search_result.success:
        logger.info(f"Jina API returned {len(search_result.scraped_pages)} pages")
    else:
        logger.warning(f"Failure to call Jina API")

    logger.info(f"Number of burned Jina API tokens: {search_result.total_used_tokens}")

    return search_result.model_dump()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=True,
                markup=True
            )
        ]
    )

    query = input("Enter your query for web search:\n")
    result = search_web_formatted_str_out(query)

    with open("jina_output.txt", "w") as f:
        f.write('\n\n'.join(result))

    # logger.info(f"Web search results: {'\n\n'.join(result)}")
