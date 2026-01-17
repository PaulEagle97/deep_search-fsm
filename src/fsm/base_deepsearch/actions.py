import logging
from typing import Optional

from burr.core import action

from haystack.dataclasses import ChatMessage, ChatRole, StreamingCallbackT

from ...nlp import build_openai_generator_pipe
from ...tools import init_tool_invoker
from .models import ApplicationState
from .config import CURRENT_TOOLS
from .prompt import get_sys_prompt

logger = logging.getLogger(__name__)


@action.pydantic(
    reads=[],
    writes=[
        "chat_history",
    ],
)
def build_chat_msgs(
    state: ApplicationState,
    query: Optional[str] = None,
) -> ApplicationState:
    if not query:
        query = input("Type your question:\n")

    user_message = ChatMessage.from_user(query)
    sys_message = ChatMessage.from_system(get_sys_prompt())

    state.chat_history.extend([sys_message, user_message])
    return state


@action.pydantic(
    reads=[
        "chat_history",
        "counter",
    ],
    writes=[
        "chat_history",
        "counter",
    ],
)
def ai_response(
    state: ApplicationState,
    max_iterations: int,
    streaming_callback: Optional[StreamingCallbackT] = None,
) -> ApplicationState:
    generator_pipe, input_builder, output_parser = build_openai_generator_pipe()
    pipe_input = input_builder(
                    msgs=state.chat_history,
                    generator_run_kwargs={
                        "streaming_callback": streaming_callback,
                        "generation_kwargs": {
                            "tool_choice": "none" if state.counter >= max_iterations else "auto",
                            "reasoning_effort": "minimal",
                        },
                        "tools": CURRENT_TOOLS,
                        "tools_strict": True,

                    },
                )
    pipe_output = generator_pipe.run(
        pipe_input,
    )
    ass_msg: ChatMessage = output_parser(pipe_output)[0]
    state.chat_history.append(ass_msg)
    state.counter += 1

    return state


@action.pydantic(
    reads=[
        "chat_history",
    ],
    writes=[
        "chat_history",
        "should_continue",
    ],
)
def tool_invocation(
    state: ApplicationState,
) -> ApplicationState:

    ass_msg: ChatMessage = state.chat_history[-1]
    if ass_msg.role != ChatRole.ASSISTANT:
        raise ValueError("Last chat message must be an assistant message")

    if not ass_msg.tool_calls:
        state.should_continue = False
        return state

    tool_invoker = init_tool_invoker(CURRENT_TOOLS)
    tool_invoker_result = tool_invoker.run(
        messages=[ass_msg]
    )
    tool_messages = tool_invoker_result["tool_messages"]

    state.chat_history.extend(tool_messages)
    return state


@action.pydantic(
    reads=[
        "counter",
    ],
    writes=[
    ],
)
def end(
    state: ApplicationState,
) -> ApplicationState:
    logger.info(f"FSM finished after {state.counter} iterations")

    return state
