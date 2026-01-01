from typing import Any, List, Dict, Annotated, Optional

from haystack.tools import create_tool_from_function
from haystack.components.tools import ToolInvoker


def search_web(
    query: Annotated[str, "A query to be searched in the web"],
) -> List[Optional[str]]:
    """
    Performs a web search and returns retrieved web page texts.
    """
    # TODO: implement
    ...


web_search_tool = create_tool_from_function(
    function=search_web
)

CURRENT_TOOLS = [web_search_tool]


def init_tool_invoker(
    raise_on_tool_invocation_failure: bool = False,
    tool_invoker_kwargs: Optional[Dict[str, Any]] = None,
) -> ToolInvoker:
    resolved_tool_invoker_kwargs = {
        "tools": CURRENT_TOOLS,
        "raise_on_failure": raise_on_tool_invocation_failure,
        **(tool_invoker_kwargs or {}),
    }
    return ToolInvoker(**resolved_tool_invoker_kwargs)
