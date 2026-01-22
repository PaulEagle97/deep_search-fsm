from typing import Dict, Tuple, Callable, List, Any

from pydantic import BaseModel

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
        azure_deployment=azure_config.DEPLOYMENT_NAME,
    )

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")

    def input(
        msgs: List[ChatMessage],
        generator_run_kwargs: Dict[str, Any],
        template_variables: Dict[str, Any] | None = None,
    ) -> Dict:
        return {
            "prompt_builder": {
                "template": msgs,
                **({"template_variables": template_variables} if template_variables else {}),
            },
            "llm": {**generator_run_kwargs},
        }

    def output(response: Dict[str, Any]) -> List[ChatMessage]:
        return response["llm"]["replies"]

    return pipe, input, output


def build_struct_generator_pipe() -> Tuple[Pipeline, Callable, Callable]:
    prompt_builder = ChatPromptBuilder()
    llm = AzureOpenAIChatGenerator(
        api_key=Secret.from_token(azure_config.OPENAI_API_KEY),
        azure_endpoint=azure_config.OPENAI_ENDPOINT,
        azure_deployment=azure_config.DEPLOYMENT_NAME,
    )

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")

    def input(
        msgs: List[ChatMessage],
        struct_model: BaseModel,
        generator_run_kwargs: Dict[str, Any] = {},
        template_variables: Dict[str, Any] | None = None,
        strict: bool = True,
    ) -> Dict:
        pydantic_schema = struct_model.model_json_schema()
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": pydantic_schema["title"],
                "description": pydantic_schema.get("description", ""),
                "schema": {
                    "type": "object",
                    "properties": pydantic_schema["properties"],
                    "required": pydantic_schema.get("required", []),
                    **({"additionalProperties": False} if strict else {}),
                },
                "strict": strict
            }
        }

        return {
            "prompt_builder": {
                "template": msgs,
                **({"template_variables": template_variables} if template_variables else {}),
            },
            "llm": {
                **generator_run_kwargs,
                "generation_kwargs": {
                    **generator_run_kwargs.get("generation_kwargs", {}),
                    "response_format": json_schema
                }
            },
        }

    def output(response: Dict[str, Any]) -> List[ChatMessage]:
        return response["llm"]["replies"]

    return pipe, input, output
