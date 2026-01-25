import logging
from typing import Optional

from burr.core import action

from haystack.dataclasses import ChatMessage

from ...models import SearchReasoning
from ...nlp import build_struct_generator_pipe
from ...tools import jina_search, jina_result_to_formatted_pages

from .models import ApplicationState
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
        "search_results",
        "token_counter",
        "search_counter",
        "msg_history",
    ],
)
def invoke_web_search_tool(
    state: ApplicationState,
) -> ApplicationState:
    logger.info(f"Calling Jina API with query='{state.next_search_query}'")
    search_result = jina_search(state.next_search_query)

    if search_result.success:
        logger.info(f"Jina API returned {len(search_result.scraped_pages)} pages")
    else:
        logger.warning(f"Failure to call Jina API")

    logger.info(f"Number of burned Jina API tokens: {search_result.total_used_tokens}")

    state.search_results.append(search_result)
    state.token_counter += search_result.total_used_tokens
    state.search_counter += 1
    # this message will be swapped
    state.msg_history.append(
        ChatMessage.from_user(
            get_iterative_web_results_user_prompt_template()
        )
    )

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
        },
    )
    pipe_output = struct_pipe.run(
        pipe_input,
    )
    ass_msg: ChatMessage = output_parser(pipe_output)[0]

    llm_reasoning = SearchReasoning.model_validate_json(ass_msg.text)
    state.next_search_query = llm_reasoning.next_search_query

    # we remove content from the search results to save tokens
    state.msg_history[-1] = ChatMessage.from_user(get_iterative_web_results_user_prompt("\n---\n".join(pages_without_content)))
    state.msg_history.append(ass_msg)

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
