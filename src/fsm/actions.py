import logging
from typing import Optional

from burr.core import action

from haystack.dataclasses import ChatMessage, ChatRole, StreamingCallbackT

from ..models import ApplicationState
from ..nlp import build_openai_generator_pipe
from ..tools import CURRENT_TOOLS, init_tool_invoker

logger = logging.getLogger(__name__)


@action.pydantic(
    reads=[],
    writes=[
        "chat_history",
    ],
)
def human_input(
    state: ApplicationState,
    query: Optional[str] = None,
) -> ApplicationState:
    if not query:
        query = input("Type your question:\n")

    user_message = ChatMessage.from_user(query)

    state.chat_history.append(user_message)
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

    tool_invoker = init_tool_invoker()
    tool_invoker_result = tool_invoker.run(
        messages=[ass_msg]
    )
    tool_messages = tool_invoker_result["tool_messages"]

    state.chat_history.extend(tool_messages)
    return state


@action.pydantic(
    reads=[
        "counter",
        "chat_history"
    ],
    writes=[
    ],
)
def end(
    state: ApplicationState,
) -> ApplicationState:
    logger.info(f"FSM finished after {state.counter} iterations")

    return state
