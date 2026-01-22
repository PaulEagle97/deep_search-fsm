import yaml
from pathlib import Path

from pydantic import BaseModel

from haystack.tools import create_tool_from_function

from ...tools import search_web_structured_out


class FSMConfig(BaseModel):
    LLM_ITERATIONS_THRESHOLD: int
    SEARCH_CONTEXT_TOKEN_LIMIT: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FSMConfig":
        p = Path(path)
        if p.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("The file must have a YAML extension")

        data = yaml.safe_load(p.read_text(encoding="utf-8"))

        return cls.model_validate(data)


_config_path = Path(__file__).parent / "config.yaml"
fsm_config = FSMConfig.from_yaml(_config_path)

web_search_tool = create_tool_from_function(
    search_web_structured_out,
    name="web_search",
    description="Performs a web search and returns scraped web page summaries."
)

CURRENT_TOOLS = [web_search_tool]
