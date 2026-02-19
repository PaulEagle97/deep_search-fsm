# Deep Search FSM

Agentic research agents built with finite-state machines. Given a research query, the agents perform iterative web search, synthesize sources, and produce structured reports.

## Overview

Deep Search FSM implements two research agent variants:

- **base_deepsearch** — A simple agent that uses tool-calling LLMs to alternate between reasoning and web search until a stopping condition.
- **v1_deepsearch** — An iterative agent that runs multiple search rounds, evaluates results, decides on follow-up queries, selects and trims sources, then generates a final report.

Both are implemented as [Burr](https://github.com/dagster-io/burr) applications with explicit state machines.

## Features

- **Iterative web search** — Agents refine queries across multiple rounds based on prior results.
- **Source synthesis** — Content from [Jina Reader](https://jina.ai/reader/) is token-counted, trimmed, and used for report generation.
- **Multi-provider LLM support** — Azure OpenAI for reasoning and structured output; Google Gemini for report generation and token counting.
- **Per-app configuration** — Model selection and limits live in each app's `config.yaml`.

## Requirements

- Python 3.12
- [PDM](https://pdm-project.org/) for dependency management

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/deep_search-fsm.git
cd deep_search-fsm

# Install dependencies with PDM
pdm install
```

## Configuration

### Environment variables

Copy the example env files and fill in your API keys:

```bash
cp openai.env.example openai.env
cp azure.env.example azure.env
cp google.env.example google.env
cp jina.env.example jina.env
```

| File        | Purpose                                      |
|-------------|----------------------------------------------|
| `openai.env` | OpenAI API key (optional, for OpenAI pipes)  |
| `azure.env`  | Azure OpenAI API key and endpoint            |
| `google.env` | Google Gemini API key                       |
| `jina.env`   | Jina Reader API key for web search           |

### App configuration

Each app has its own `config.yaml` in `src/fsm/<app>/`:

**base_deepsearch** (`src/fsm/base_deepsearch/config.yaml`):
- `LLM_ITERATIONS_THRESHOLD` — Max reasoning iterations before forcing final answer
- `AZURE_DEPLOYMENT` — Azure OpenAI deployment name

**v1_deepsearch** (`src/fsm/v1_deepsearch/config.yaml`):
- `MAX_NUMBER_SEARCHES` — Max search rounds
- `SEARCH_TOKEN_LIMIT` — Token budget per search result
- `SOURCES_TOKEN_LIMIT` — Total token budget for report sources
- `AZURE_DEPLOYMENT` — Azure deployment for structured output (search reasoning)
- `GEMINI_MODEL` — Gemini model for report generation and token counting

## Usage

### Run v1_deepsearch (recommended)

```bash
pdm run python -m src.fsm.v1_deepsearch.app
```

The app uses the query defined in the script and writes the report to `research_report.md` in the current directory.

### Run base_deepsearch

```bash
pdm run python -m src.fsm.base_deepsearch.app
```

Output is written to `research_result.md`.

### Visualize the state machine

```python
from src.fsm.v1_deepsearch.app import build_burr_app

app = build_burr_app(visualize=True)
# Generates v1_deepsearch_app.png
```

## Project structure

```
deep_search-fsm/
├── src/
│   ├── core/           # Shared config (Jina, env loaders)
│   ├── fsm/
│   │   ├── base_deepsearch/   # Simple tool-calling agent
│   │   └── v1_deepsearch/     # Iterative search + report agent
│   ├── models/         # Pydantic models, LLM config
│   ├── nlp/            # LLM pipes (Azure, Gemini), tokenizers
│   └── tools/          # Jina search, tool invoker
├── openai.env.example
├── azure.env.example
├── google.env.example
├── jina.env.example
├── pyproject.toml
└── README.md
```

## Dependencies

- [Burr](https://github.com/dagster-io/burr) — State machine framework
- [Haystack](https://haystack.deepset.ai/) — LLM pipelines and generators
- [Jina Reader](https://jina.ai/reader/) — Web search and content extraction
- [google-genai](https://github.com/google/genai-python) — Gemini API

## License

MIT
