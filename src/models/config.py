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
