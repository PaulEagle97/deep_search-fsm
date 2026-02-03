import yaml
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    API_KEY: str

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file="openai.env",
    )


class AzureOpenAISettings(BaseSettings):
    DEPLOYMENT_NAME: str
    OPENAI_API_KEY: str
    OPENAI_ENDPOINT: str

    model_config = SettingsConfigDict(
        env_prefix="AZURE_",
        env_file="azure.env"
    )


class JinaEnvs(BaseSettings):
    API_KEY: str

    model_config = SettingsConfigDict(
        env_prefix="JINA_",
        env_file="jina.env",
    )


class GeminiSettings(BaseSettings):
    API_KEY: str
    MODELS: list[str]

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file="google.env",
    )


class JinaConfig(BaseModel):
    envs: JinaEnvs = Field(default_factory=JinaEnvs)
    NUM_PAGES_PER_SEARCH: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JinaConfig":
        p = Path(path)
        if p.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("The file must have a YAML extension")

        data = yaml.safe_load(p.read_text(encoding="utf-8"))

        return cls.model_validate(data)
