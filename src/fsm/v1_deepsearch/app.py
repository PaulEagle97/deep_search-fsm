import json
import logging

from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from burr.core import Application, ApplicationBuilder, when
from burr.integrations.pydantic import PydanticTypingSystem

from haystack.dataclasses import ChatRole

from .models import ApplicationState
from .actions import (
    init_msg_history,
    loop_breaker,
    generate_search_params,
    invoke_web_search_tool,
    end,
)
from .config import fsm_config

logger = logging.getLogger(__name__)


def build_burr_app(visualize: bool = False) -> Application:
    app = (
        ApplicationBuilder()
        .with_actions(
            init_msg_history,
            invoke_web_search_tool,
            loop_breaker.bind(
                max_iterations=fsm_config.LLM_ITERATIONS_THRESHOLD,
                token_limit=fsm_config.SEARCH_CONTEXT_TOKEN_LIMIT,
            ),
            generate_search_params,
            end,
        )
        .with_transitions(
            ("init_msg_history", "invoke_web_search_tool"),
            ("invoke_web_search_tool", "loop_breaker"),
            ("loop_breaker", "generate_search_params", when(continue_search=True)),
            ("loop_breaker", "end", when(continue_search=False)),
            ("generate_search_params", "invoke_web_search_tool"),
        )
        .with_typing(PydanticTypingSystem(ApplicationState))
        .with_state(ApplicationState())
        .with_entrypoint("init_msg_history")
        .with_tracker(project="v1_deepsearch")
        .build()
    )

    if visualize:
        app.visualize(
            output_file_path="./v1_deepsearch_app.png",
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
    query = "From 2020 to 2050, how many elderly people will there be in Japan? What is their consumption potential across various aspects such as clothing, food, housing, and transportation? Based on population projections, elderly consumer willingness, and potential changes in their consumption habits, please produce a market size analysis report for the elderly demographic."
    query = "I am looking for Ausbildung and Studium programs in the UI/UX domain. I live in Rosenheim, Germany. Research what current options are available for me to enroll."
    query = "How to build a house in Germany? Explore the whole process from city approval to buying materials and finding contractors."
    final_action, result, state = app.run(
        halt_after=["end"], 
        inputs={"query" : query}
    )

    typed_state: ApplicationState = state.data
    console = Console()

    # Print message history with proper formatting
    for i, msg in enumerate(typed_state.msg_history):
        role_style = {
            ChatRole.SYSTEM: "bold magenta",
            ChatRole.USER: "bold cyan",
            ChatRole.ASSISTANT: "bold green",
        }.get(msg.role, "white")

        content = msg.text or "[empty]"
        if msg.role == ChatRole.ASSISTANT:
            parsed = json.loads(content)
            formatted_json = json.dumps(parsed, indent=2, ensure_ascii=False)
            content = Syntax(formatted_json, "json", theme="ansi_light", word_wrap=True)

        console.print(Panel(
            content,
            title=f"Message {i} | {msg.role}",
            title_align="left",
            border_style=role_style,
        ))

    # Save search results to markdown
    with open("search_results.md", "w") as f:
        f.write("# Iterative Search Results\n\n")
        f.write(f"**User Query:** {typed_state.user_query}\n\n")
        f.write(f"**Total Iterations:** {typed_state.search_counter}\n\n")
        f.write(f"**Total Tokens:** {typed_state.token_counter}\n\n")
        f.write("---\n\n")
        
        for iteration_idx, search_result in enumerate(typed_state.search_results):
            f.write(f"## Iteration {iteration_idx + 1}: `{search_result.query}`\n\n")
            f.write(f"**Success:** {search_result.success} | **Tokens:** {search_result.total_used_tokens}\n\n")
            
            for page_idx, page in enumerate(search_result.scraped_pages):
                f.write(f"### Page {page_idx + 1}: {page.title}\n\n")
                f.write(f"**URL:** {page.url}\n\n")
                f.write(f"**Description:** {page.description}\n\n")
                f.write(f"**Content:**\n\n```\n{page.content[:2000]}{'...' if len(page.content) > 2000 else ''}\n```\n\n")
                f.write("---\n\n")
    
    logger.info(f"Search results saved to search_results.md")
