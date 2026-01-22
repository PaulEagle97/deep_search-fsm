import logging
from typing import Optional

from burr.core import action

from haystack.dataclasses import ChatMessage, ChatRole

from ...models import JinaReaderSearchResult, PageEvaluation
from ...nlp import build_openai_generator_pipe, build_struct_generator_pipe
from ...tools import init_tool_invoker

from .models import ApplicationState
from .config import CURRENT_TOOLS, web_search_tool
from .utils import build_page_evaluation_msgs, page_is_good_enough, build_tool_msg_from_evals
from .prompt import get_iterative_searcher_sys_prompt, get_iterative_searcher_user_prompt_template

logger = logging.getLogger(__name__)


@action.pydantic(
    reads=[],
    writes=[
        "user_query",
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

    sys_message = ChatMessage.from_system(get_iterative_searcher_sys_prompt())
    user_message = ChatMessage.from_user(get_iterative_searcher_user_prompt_template())

    state.msg_history.extend([sys_message, user_message])
    return state


@action.pydantic(
    reads=[
        "msg_history",
        "user_query"
    ],
    writes=[
        "msg_history",
    ],
)
def generate_search_params(
    state: ApplicationState,
) -> ApplicationState:
    generator_pipe, input_builder, output_parser = build_openai_generator_pipe()
    pipe_input = input_builder(
                    msgs=state.msg_history,
                    generator_run_kwargs={
                        "generation_kwargs": {
                            "tool_choice": {
                                "type": "function",
                                "function": {"name": web_search_tool.name}
                            },
                            "reasoning_effort": "low",
                            "parallel_tool_calls": False,
                        },
                        "tools": CURRENT_TOOLS,
                        "tools_strict": True,
                    },
                    template_variables={
                        "user_query": state.user_query,
                    },
                )
    pipe_output = generator_pipe.run(
        pipe_input,
    )
    ass_msg: ChatMessage = output_parser(pipe_output)[0]
    state.msg_history.append(ass_msg)

    return state


@action.pydantic(
    reads=[
        "msg_history",
    ],
    writes=[
        "search_results",
        "search_counter",
    ],
)
def invoke_web_search_tool(
    state: ApplicationState,
) -> ApplicationState:
    ass_msg: ChatMessage = state.msg_history[-1]
    if ass_msg.role != ChatRole.ASSISTANT:
        raise ValueError("Last message must be an assistant message")

    tool_invoker = init_tool_invoker(
        CURRENT_TOOLS,
        tool_invoker_kwargs={
            "convert_result_to_json_string": True,
        }
    )
    tool_invoker_result = tool_invoker.run(
        messages=[ass_msg]
    )
    tool_messages = tool_invoker_result["tool_messages"]

    if not all([msg.tool_call_result for msg in tool_messages]):
        raise ValueError("Every tool message must have tool call result")

    state.search_results.append(
        [
            JinaReaderSearchResult.model_validate_json(
                msg.tool_call_result.result
            ) for msg in tool_messages
        ]
    )
    state.search_counter += 1

    return state


@action.pydantic(
    reads=[
        "search_results",
    ],
    writes=[
        "search_results",
    ],
)
def evaluate_pages(
    state: ApplicationState,
) -> ApplicationState:
    for result in state.search_results[-1]:
        for idx, page in enumerate(result.scraped_pages, 1):
            struct_pipe, input_builder, output_parser = build_struct_generator_pipe()
            pipe_input = input_builder(
                msgs=build_page_evaluation_msgs(),
                struct_model=PageEvaluation,
                template_variables={
                    "search_query": result.query,
                    "page_content": page.content,
                },
            )
            pipe_output = struct_pipe.run(
                pipe_input,
            )
            ass_msg: ChatMessage = output_parser(pipe_output)[0]

            page.llm_eval = PageEvaluation.model_validate_json(ass_msg.text)
            
            # Print scores for this page
            title_short = page.title[:40] + "..." if len(page.title) > 40 else page.title
            logger.info(f"  [green]Page {idx}:[/green] depth={page.llm_eval.depth_score}/5, rel={page.llm_eval.relevance_score}/5 | {title_short}")

    return state


@action.pydantic(
    reads=[
        "search_results",
    ],
    writes=[
        "search_results",
    ],
)
def filter_low_quality_pages(
    state: ApplicationState,
) -> ApplicationState:
    # Filter pages in the latest iteration
    for result in state.search_results[-1]:
        result.scraped_pages = [
            page for page in result.scraped_pages
            if page_is_good_enough(page)
        ]
    
    # Log how many pages remain
    total_pages = sum(len(r.scraped_pages) for r in state.search_results[-1])
    logger.info(f"Kept {total_pages} high-quality pages after filtering")

    return state


@action.pydantic(
    reads=[
        "search_results",
        "msg_history",
    ],
    writes=[
        "msg_history",
    ],
)
def llm_context_builder(
    state: ApplicationState,
) -> ApplicationState:
    filtered_pages = [
        page
        for result in state.search_results[-1]
        for page in result.scraped_pages
    ]

    ass_msg: ChatMessage = state.msg_history[-1]
    if ass_msg.role != ChatRole.ASSISTANT:
        raise ValueError("Last message must be an assistant message")
    web_search_tool_call = ass_msg.tool_call

    web_search_tool_msg = build_tool_msg_from_evals(filtered_pages, web_search_tool_call)
    state.msg_history.append(web_search_tool_msg)

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
        "search_counter",
    ],
    writes=[
    ],
)
def end(
    state: ApplicationState,
) -> ApplicationState:
    logger.info(f"FSM finished after {state.search_counter} iterations")

    return state
