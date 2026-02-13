import yaml
from pathlib import Path

from pydantic import BaseModel


class FSMConfig(BaseModel):
    MAX_NUMBER_SEARCHES: int
    SEARCH_TOKEN_LIMIT: int
    SOURCES_TOKEN_LIMIT: int
    AZURE_DEPLOYMENT: str
    GEMINI_MODEL: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FSMConfig":
        p = Path(path)
        if p.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("The file must have a YAML extension")

        data = yaml.safe_load(p.read_text(encoding="utf-8"))

        return cls.model_validate(data)


_config_path = Path(__file__).parent / "config.yaml"
fsm_config = FSMConfig.from_yaml(_config_path)
