import tiktoken

_encoder = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))
