from pathlib import Path

from ..models import AzureOpenAISettings, JinaConfig, GeminiSettings

azure_config = AzureOpenAISettings()
gemini_config = GeminiSettings()

jina_config_path = Path(__file__).parent / "jina.yaml"
jina_config = JinaConfig.from_yaml(jina_config_path)
