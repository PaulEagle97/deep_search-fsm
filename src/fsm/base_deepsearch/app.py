import ast
import logging
from textwrap import shorten
from rich.logging import RichHandler

from burr.core import Application, ApplicationBuilder, when
from burr.integrations.pydantic import PydanticTypingSystem
from haystack.components.generators.utils import print_streaming_chunk

from .models import ApplicationState
from .actions import (
    build_chat_msgs,
    ai_response,
    tool_invocation,
    end,
)
from .config import fsm_config

logger = logging.getLogger(__name__)


def build_burr_app(visualize: bool = False) -> Application:
    app = (
        ApplicationBuilder()
        .with_actions(
            build_chat_msgs,
            ai_response.bind(
                max_iterations=fsm_config.LLM_ITERATIONS_THRESHOLD,
                # streaming_callback=print_streaming_chunk,
            ),
            tool_invocation,
            end,
        )
        .with_transitions(
            ("build_chat_msgs", "ai_response"),
            ("ai_response", "tool_invocation"),
            ("tool_invocation", "ai_response", when(should_continue=True)),
            ("tool_invocation", "end", when(should_continue=False)),
        )
        .with_typing(PydanticTypingSystem(ApplicationState))
        .with_state(ApplicationState())
        .with_entrypoint("build_chat_msgs")
        .with_tracker(project="base_deepsearch")
        .build()
    )

    if visualize:
        app.visualize(
            output_file_path="./base_deepsearch_app.png",
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
        inputs={"query" : "What is current SOTA in agentic frameworks?"}
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
            truncated = shorten(str(result), width=1000, placeholder=" ...")
            logger.info(f"  Tool Result: {truncated}")
        logger.info("---")

    with open("research_result.md", "w") as f:
        f.write(typed_state.chat_history[-1].text)
