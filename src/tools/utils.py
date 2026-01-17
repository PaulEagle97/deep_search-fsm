from typing import Any, List, Dict, Optional

from haystack.tools import Tool
from haystack.components.tools import ToolInvoker


def init_tool_invoker(
    tools: List[Tool],
    tool_invoker_kwargs: Optional[Dict[str, Any]] = None,
) -> ToolInvoker:
    resolved_tool_invoker_kwargs = {
        "tools": tools,
        **(tool_invoker_kwargs or {}),
    }
    return ToolInvoker(**resolved_tool_invoker_kwargs)
