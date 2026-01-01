from typing import Dict, Tuple, Callable, List, Any

from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import AzureOpenAIChatGenerator


from ..core import azure_config


def build_openai_generator_pipe() -> Tuple[Pipeline, Callable, Callable]:
    prompt_builder = ChatPromptBuilder()
    llm = AzureOpenAIChatGenerator(
        api_key=Secret.from_token(azure_config.OPENAI_API_KEY),
        azure_endpoint=azure_config.OPENAI_ENDPOINT,
        azure_deployment="gpt-4o-mini-service",
    )

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")

    def input(
        msgs: List[ChatMessage],
        generator_run_kwargs: Dict[str, Any],
    ) -> Dict:
        return {
            "prompt_builder": {"template": msgs},
            "llm": {**generator_run_kwargs},
        }

    def output(response: Dict[str, Any]) -> List[ChatMessage]:
        return response["llm"]["replies"]

    return pipe, input, output
