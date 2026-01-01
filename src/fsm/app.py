import ast
import logging
from rich.logging import RichHandler

from burr.core import Application, ApplicationBuilder, when
from burr.integrations.pydantic import PydanticTypingSystem
from haystack.components.generators.utils import print_streaming_chunk


from ..models import ApplicationState
from .actions import (
    human_input,
    ai_response,
    tool_invocation,
    end,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


def build_burr_app(visualize: bool = False) -> Application:
    app = (
        ApplicationBuilder()
        .with_actions(
            human_input,
            ai_response.bind(
                max_iterations=MAX_ITERATIONS,
                # streaming_callback=print_streaming_chunk,
            ),
            tool_invocation,
            end,
        )
        .with_transitions(
            ("human_input", "ai_response"),
            ("ai_response", "tool_invocation"),
            ("tool_invocation", "ai_response", when(should_continue=True)),
            ("tool_invocation", "end", when(should_continue=False)),
        )
        .with_typing(PydanticTypingSystem(ApplicationState))
        .with_state(ApplicationState())
        .with_entrypoint("human_input")
        .with_tracker(project="demo_getting_started")
        .build()
    )

    if visualize:
        app.visualize(
            output_file_path="./burr_app.png",
            include_conditions=True,
        )

    return app


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

    app = build_burr_app()
    final_action, result, state = app.run(
        halt_after=["end"], 
        # inputs={"query" : "How to register a marriage? How to register a dog? How to pay a real estate tax?"}
    )

    typed_state: ApplicationState = state.data

    for i, msg in enumerate(typed_state.chat_history):
        logger.info(f"Message {i}:")
        logger.info(f"  Role: {msg.role}")
        logger.info(f"  Content: {msg.text}")
        if msg.tool_calls:
            logger.info(f"  Tool Calls: {msg.tool_calls}")
        if msg.tool_call_result:
            result = ast.literal_eval(msg.tool_call_result.result)
            logger.info(f"  Tool Result: {result}")
        logger.info("---")
