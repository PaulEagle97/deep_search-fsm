import logging

from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel

from burr.core import Application, ApplicationBuilder, when
from burr.integrations.pydantic import PydanticTypingSystem

from haystack.dataclasses import ChatRole

from .models import ApplicationState
from .actions import (
    init_msg_history,
    loop_breaker,
    generate_search_params,
    invoke_web_search_tool,
    prepare_report_sources,
    generate_report,
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
            prepare_report_sources.bind(
                token_limit=fsm_config.SEARCH_CONTEXT_TOKEN_LIMIT,
            ),
            generate_report,
            end,
        )
        .with_transitions(
            ("init_msg_history", "invoke_web_search_tool"),
            ("invoke_web_search_tool", "loop_breaker"),
            ("loop_breaker", "generate_search_params", when(continue_search=True)),
            ("loop_breaker", "prepare_report_sources", when(continue_search=False)),
            ("generate_search_params", "invoke_web_search_tool"),
            ("prepare_report_sources", "generate_report"),
            ("generate_report", "end"),
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
    query = "How to build a house in Germany? Explore the whole process from city approval to buying materials and finding contractors."
    query = "I am looking for Ausbildung and Studium programs in the UI/UX domain. I live in Rosenheim, Germany. Research what current options are available for me to enroll."
    query = "How to build a house in Germany? Explain the whole process."
    query = "What are the investment philosophies of Duan Yongping, Warren Buffett, and Charlie Munger?"
    query = "From 2020 to 2050, how many elderly people will there be in Japan? What is their consumption potential across various aspects such as clothing, food, housing, and transportation? Based on population projections, elderly consumer willingness, and potential changes in their consumption habits, please produce a market size analysis report for the elderly demographic."
    query = "Write a research paper about SOTA in Deep Search agentic systems with practical examples."
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

        console.print(
            Panel(
                content,
                title=f"Message {i} | {msg.role}",
                title_align="left",
                border_style=role_style,
            )
        )

    # Save research report to markdown
    with open("research_report.md", "w") as f:
        # Header
        f.write("# Research Report\n\n")
        
        # Research Task
        f.write("## Research Task\n\n")
        f.write(f"{typed_state.user_query}\n\n")
        f.write("---\n\n")
        
        # Final Report
        f.write("## Report\n\n")
        f.write(f"{typed_state.final_report}\n\n")
        f.write("---\n\n")
        
        # Search Queries
        f.write("## Search Queries\n\n")
        f.write(f"*{len(typed_state.executed_queries)} queries executed across {typed_state.search_counter} iterations*\n\n")
        for idx, query in enumerate(typed_state.executed_queries, 1):
            f.write(f"{idx}. `{query}`\n")
        f.write("\n---\n\n")
        
        # Report Sources
        f.write("## Sources\n\n")
        f.write(f"*{len(typed_state.report_sources)} sources used for report generation*\n\n")
        for idx, page in enumerate(typed_state.report_sources, 1):
            f.write(f"### [{idx}] {page.title}\n\n")
            f.write(f"**URL:** {page.url}\n\n")
            f.write(f"**Description:** {page.description}\n\n")
            content_preview = page.content[:3000]
            if len(page.content) > 3000:
                content_preview += "..."
            f.write(f"<details>\n<summary>Content preview ({page.content_tokens} tokens)</summary>\n\n```\n{content_preview}\n```\n\n</details>\n\n")
    
    logger.info("Research report saved to research_report.md")
