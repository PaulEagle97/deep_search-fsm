import logging
from typing import Optional

from burr.core import action

from haystack.dataclasses import ChatMessage

from ...models import SearchReasoningNextQuery
from ...nlp import build_azure_openai_struct_pipe, build_gemini_chat_pipe, count_gemini_tokens
from ...tools import jina_search, jina_result_to_formatted_pages

from .models import ApplicationState
from .config import fsm_config
from .utils import format_llm_reasoning_next_query, format_pages_for_report, build_iterative_searcher_msgs, build_report_generator_msgs, count_content_tokens, trim_content_tokens
from .prompt import get_iterative_web_results_user_prompt_template, get_iterative_web_results_user_prompt

logger = logging.getLogger(__name__)


@action.pydantic(
    reads=[],
    writes=[
        "user_query",
        "next_search_query",
        "msg_history",
    ],
)
def init_msg_history(
    state: ApplicationState,
    query: Optional[str] = None,
) -> ApplicationState:
    if not query:
        query = input("Type your question:\n")

    state.user_query = query
    state.next_search_query = query
    state.msg_history = build_iterative_searcher_msgs()

    return state


@action.pydantic(
    reads=[
        "next_search_query",
    ],
    writes=[
        "executed_queries",
        "search_results",
        "sources_token_counter",
        "search_counter",
    ],
)
def invoke_web_search_tool(
    state: ApplicationState,
    search_token_limit: int,
) -> ApplicationState:
    logger.info(f"Calling Jina API with query='{state.next_search_query}'")
    search_result = jina_search(state.next_search_query)

    logger.info(f"Burned Jina API tokens: {search_result.total_jina_tokens}")
    logger.info(f"Jina API returned {len(search_result.scraped_pages)} pages")

    tokenizer = lambda t: count_gemini_tokens(t, fsm_config.GEMINI_MODEL)
    total_tokens, search_result = count_content_tokens(search_result, tokenizer)
    logger.info(f"Counted {total_tokens} page content tokens ({search_token_limit} allowed)")

    trimmed_tokens, search_result = trim_content_tokens(search_result, tokenizer, search_token_limit)
    logger.info(f"{trimmed_tokens} tokens left after trimming ({len(search_result.scraped_pages)} pages)")

    state.search_results.append(search_result)
    state.executed_queries.append(state.next_search_query)
    state.sources_token_counter += trimmed_tokens
    state.search_counter += 1

    return state


@action.pydantic(
    reads=[
        "search_counter",
        "sources_token_counter",
    ],
    writes=[
        "continue_search",
    ],
)
def loop_breaker(
    state: ApplicationState,
    max_searches: int,
    sources_token_limit: int,
) -> ApplicationState:
    logger.info(f"Finished search N.{state.search_counter}, {state.sources_token_counter} source tokens accumulated so far")

    if state.search_counter >= max_searches or state.sources_token_counter >= sources_token_limit:
        state.continue_search = False
    
    return state


@action.pydantic(
    reads=[
        "search_results",
        "msg_history",
        "user_query",
        "executed_queries",
    ],
    writes=[
        "next_search_query",
        "msg_history",
    ],
)
def generate_search_params(
    state: ApplicationState,
) -> ApplicationState:
    if not state.search_results:
        raise ValueError("There must be at least one search result")
    pages_with_content = jina_result_to_formatted_pages(state.search_results[-1])
    pages_without_content = jina_result_to_formatted_pages(state.search_results[-1], include_content=False)

    # this message will be swapped
    state.msg_history.append(
        ChatMessage.from_user(
            get_iterative_web_results_user_prompt_template()
        )
    )

    struct_pipe, input_builder, output_parser = build_azure_openai_struct_pipe(fsm_config.AZURE_DEPLOYMENT)
    pipe_input = input_builder(
        msgs=state.msg_history,
        struct_model=SearchReasoningNextQuery,
        generator_run_kwargs={
            "generation_kwargs": {
                # "reasoning_effort": "low",
            },
        },
        template_variables={
            "user_query": state.user_query,
            "search_result": "\n---\n".join(pages_with_content),
            "executed_queries": state.executed_queries,
        },
    )
    pipe_output = struct_pipe.run(
        pipe_input,
    )
    ass_msg: ChatMessage = output_parser(pipe_output)[0]
    llm_reasoning = SearchReasoningNextQuery.model_validate_json(ass_msg.text)

    # remove content from search results to save tokens
    state.msg_history[-1] = ChatMessage.from_user(get_iterative_web_results_user_prompt("\n---\n".join(pages_without_content), state.executed_queries))
    # format JSON to LLM-friendly text and append as assistant message
    formatted_ass_msg = format_llm_reasoning_next_query(llm_reasoning)
    state.msg_history.append(ChatMessage.from_assistant(formatted_ass_msg))
    # extract next search query
    state.next_search_query = llm_reasoning.next_search_query

    return state


@action.pydantic(
    reads=[
        "search_results",
    ],
    writes=[
        "report_sources",
    ],
)
def prepare_report_sources(
    state: ApplicationState,
    sources_token_limit: int,
) -> ApplicationState:
    selected = []
    discarded = {"too_short": [], "duplicates": []}
    seen_urls = set()
    for result in state.search_results:
        for page in result.scraped_pages:
            if page.content_tokens < 500:
                discarded["too_short"].append(page)
            elif page.url in seen_urls:
                discarded["duplicates"].append(page)
            else:
                seen_urls.add(page.url)
                selected.append(page)

    token_count = sum(page.content_tokens for page in selected)

    logger.info(f"Selected {len(selected)} pages with a total of {token_count} content tokens")
    logger.info(f"Discarded {len(discarded['too_short'])} short pages and {len(discarded['duplicates'])} duplicates")

    if token_count > 0:
        token_overflow_ratio = max(0, (token_count - sources_token_limit) / token_count)
    else:
        token_overflow_ratio = 0
    logger.info(f"Every selected page's content will be trimmed by {token_overflow_ratio*100:.0f}%")
    for page in selected:
        slice_idx = int(len(page.content)*(1-token_overflow_ratio))
        page.content = page.content[:slice_idx]
        page.content_tokens = count_gemini_tokens(page.content, fsm_config.GEMINI_MODEL)

    token_count = sum(page.content_tokens for page in selected)
    logger.info(f"Total number of content tokens after trimming: {token_count}")

    state.report_sources = selected

    return state


@action.pydantic(
    reads=[
        "user_query",
        "report_sources",
    ],
    writes=[
        "final_report",
    ],
)
def generate_report(
    state: ApplicationState,
) -> ApplicationState:
    generator_pipe, input_builder, output_parser = build_gemini_chat_pipe(fsm_config.GEMINI_MODEL)
    pipe_input = input_builder(
        msgs=build_report_generator_msgs(),
        template_variables={
            "user_query": state.user_query,
            "sources": format_pages_for_report(state.report_sources),
        },
    )
    pipe_output = generator_pipe.run(pipe_input)
    ass_msg: ChatMessage = output_parser(pipe_output)[0]

    state.final_report = ass_msg.text

    return state


@action.pydantic(
    reads=[
        "search_counter",
    ],
    writes=[
    ],
)
def end(
    state: ApplicationState,
) -> ApplicationState:
    logger.info(f"FSM finished after {state.search_counter} search iterations")

    return state
