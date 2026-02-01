import logging
from typing import Optional

from burr.core import action

from haystack.dataclasses import ChatMessage

from ...models import SearchReasoning
from ...nlp import build_struct_generator_pipe
from ...tools import jina_search, jina_result_to_formatted_pages

from .models import ApplicationState
from .utils import format_llm_reasoning
from .prompt import get_iterative_searcher_sys_prompt, get_iterative_searcher_user_prompt_template, get_iterative_web_results_user_prompt_template, get_iterative_web_results_user_prompt

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

    sys_message = ChatMessage.from_system(get_iterative_searcher_sys_prompt())
    user_message = ChatMessage.from_user(get_iterative_searcher_user_prompt_template())

    state.msg_history.extend([sys_message, user_message])
    return state


@action.pydantic(
    reads=[
        "next_search_query",
    ],
    writes=[
        "executed_queries",
        "search_results",
        "token_counter",
        "search_counter",
    ],
)
def invoke_web_search_tool(
    state: ApplicationState,
) -> ApplicationState:
    logger.info(f"Calling Jina API with query='{state.next_search_query}'")
    search_result = jina_search(state.next_search_query)

    if search_result.success:
        state.executed_queries.append(state.next_search_query)
        logger.info(f"Jina API returned {len(search_result.scraped_pages)} pages")
        logger.info(f"Number of burned Jina API tokens: {search_result.total_jina_tokens}")
    else:
        logger.warning(f"Failure to call Jina API")

    state.search_counter += 1
    state.search_results.append(search_result)
    state.token_counter += sum(page.openai_tokens for page in search_result.scraped_pages)

    return state


@action.pydantic(
    reads=[
        "search_counter",
        "token_counter",
    ],
    writes=[
        "continue_search",
    ],
)
def loop_breaker(
    state: ApplicationState,
    max_iterations: int,
    token_limit: int,
) -> ApplicationState:
    if state.search_counter >= max_iterations or state.token_counter >= token_limit:
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

    struct_pipe, input_builder, output_parser = build_struct_generator_pipe()
    pipe_input = input_builder(
        msgs=state.msg_history,
        struct_model=SearchReasoning,
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
    llm_reasoning = SearchReasoning.model_validate_json(ass_msg.text)

    # remove content from search results to save tokens
    state.msg_history[-1] = ChatMessage.from_user(get_iterative_web_results_user_prompt("\n---\n".join(pages_without_content), state.executed_queries))
    # format JSON to LLM-friendly text and append as assistant message
    formatted_ass_msg = format_llm_reasoning(llm_reasoning)
    state.msg_history.append(ChatMessage.from_assistant(formatted_ass_msg))
    # extract next search query
    state.next_search_query = llm_reasoning.next_search_query

    return state


@action.pydantic(
    reads=[
        "search_results",
    ],
    writes=[
    ],
)
def filter_search_results(
    state: ApplicationState,
) -> ApplicationState:
    selected = []
    selected_token_count = 0
    discarded = {"too_short": [], "duplicates": []}
    seen_urls = set()
    for result in state.search_results:
        for page in result.scraped_pages:
            if page.openai_tokens < 500:
                discarded["too_short"].append(page)
            elif page.url in seen_urls:
                discarded["duplicates"].append(page)
            else:
                seen_urls.add(page.url)
                selected.append(page)
                selected_token_count += page.openai_tokens

    logger.info(f"Selected {len(selected)} pages with a total of {selected_token_count} OpenAI tokens")
    logger.info(f"Discarded {len(discarded["too_short"])} short pages and {len(discarded["duplicates"])} duplicates")

    return state


@action.pydantic(
    reads=[
        "search_counter",
        "token_counter",
    ],
    writes=[
    ],
)
def end(
    state: ApplicationState,
) -> ApplicationState:
    logger.info(f"FSM finished after {state.search_counter} iterations")
    logger.info(f"Total of {state.token_counter} tokens accumulated")

    return state
