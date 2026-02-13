import tiktoken
from google import genai

from ..core import gemini_config

# OpenAI tokenizer (o200k_base for GPT-4o/GPT-5)
_openai_encoder = tiktoken.get_encoding("o200k_base")

# Gemini client for token counting
_gemini_client = genai.Client(api_key=gemini_config.API_KEY)


def count_openai_tokens(text: str) -> int:
    return len(_openai_encoder.encode(text))


def count_gemini_tokens(text: str, model: str) -> int:
    result = _gemini_client.models.count_tokens(
        model=model,
        contents=text,
    )
    return result.total_tokens
