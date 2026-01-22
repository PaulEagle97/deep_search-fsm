import logging
from textwrap import shorten

from rich.logging import RichHandler

from burr.core import Application, ApplicationBuilder, when
from burr.integrations.pydantic import PydanticTypingSystem

from .models import ApplicationState
from .actions import (
    init_msg_history,
    loop_breaker,
    generate_search_params,
    invoke_web_search_tool,
    evaluate_pages,
    filter_low_quality_pages,
    llm_context_builder,
    end,
)
from .config import fsm_config

logger = logging.getLogger(__name__)


def build_burr_app(visualize: bool = False) -> Application:
    app = (
        ApplicationBuilder()
        .with_actions(
            init_msg_history,
            loop_breaker.bind(
                max_iterations=fsm_config.LLM_ITERATIONS_THRESHOLD,
                token_limit=fsm_config.SEARCH_CONTEXT_TOKEN_LIMIT,
            ),
            generate_search_params,
            invoke_web_search_tool,
            evaluate_pages,
            filter_low_quality_pages,
            llm_context_builder,
            end,
        )
        .with_transitions(
            ("init_msg_history", "generate_search_params"),
            ("generate_search_params", "invoke_web_search_tool"),
            ("invoke_web_search_tool", "evaluate_pages"),
            ("evaluate_pages", "filter_low_quality_pages"),
            ("filter_low_quality_pages", "llm_context_builder"),
            ("llm_context_builder", "loop_breaker"),
            ("loop_breaker", "generate_search_params", when(continue_search=True)),
            ("loop_breaker", "end", when(continue_search=False)),
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
    final_action, result, state = app.run(
        halt_after=["end"], 
        inputs={"query" : "Research how the world's wealthiest governments invest."}
    )

    typed_state: ApplicationState = state.data

    for i, msg in enumerate(typed_state.msg_history):
        logger.info(f"Message {i}:")
        logger.info(f"  Role: {msg.role}")
        logger.info(f"  Content: {msg.text}")
        if msg.tool_calls:
            logger.info(f"  Tool Calls: {msg.tool_calls}")
        if msg.tool_call_result:
            truncated = shorten(msg.tool_call_result.result, width=1000, placeholder=" ...")
            logger.info(f"  Tool Result: {truncated}")
        logger.info("---")

    with open("search_results.md", "w") as f:
        f.write("# Search Results with Relevance Evaluations\n\n")
        
        # search_results is List[List[JinaReaderSearchResult]]
        # - outer list = iterations
        # - inner list = search results per iteration
        for iteration_idx, iteration_results in enumerate(typed_state.search_results):
            f.write(f"## Iteration {iteration_idx + 1}\n\n")
            
            for result in iteration_results:
                f.write(f"### Search: `{result.query}`\n\n")
                f.write(f"**Success:** {result.success} | **Total Tokens:** {result.total_used_tokens}\n\n")
                
                for j, page in enumerate(result.scraped_pages):
                    f.write(f"#### Page {j + 1}: {page.title}\n\n")
                    f.write(f"**URL:** {page.url}\n\n")
                    f.write(f"**Description:** {page.description}\n\n")

                    f.write(f"##### Original Content\n\n")
                    f.write(f"```\n{page.content}\n```\n\n")

                    if page.llm_eval:
                        eval = page.llm_eval
                        f.write(f"##### LLM Evaluation\n\n")
                        f.write(f"| Dimension | Score | Summary |\n")
                        f.write(f"|-----------|-------|--------|\n")
                        f.write(f"| **Relevance** | {eval.relevance_score}/5 | {eval.relevance_summary} |\n")
                        f.write(f"| **Depth** | {eval.depth_score}/5 | {eval.depth_summary} |\n\n")
                    else:
                        f.write("*No relevance evaluation available*\n\n")
                    
                    f.write("---\n\n")
    
    logger.info(f"Search results saved to search_results.md")
