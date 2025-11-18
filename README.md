# ML Model Recommender Agent

This project uses LangChain with a self-hosted LLM (exposed via `http://127.0.0.1:11434`) to recommend machine-learning model families based on structured dataset metadata plus any extra narrative context.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) or another server compatible with the LangChain `Ollama` integration running locally at `http://127.0.0.1:11434`
- At least one model pulled (e.g., `ollama pull llama3`)
- For best compatibility, use Python 3.12 or earlier; LangChain’s legacy Pydantic v1 shims currently emit warnings on Python 3.14+

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Capture your dataset metadata as JSON. See `examples/customer_churn_metadata.json` for the expected schema.
2. Optionally craft supplemental context describing deployment constraints, KPIs, or tooling preferences.
3. Run the CLI:

```powershell
python -m src.main examples/customer_churn_metadata.json ^
  --context "Need near-real-time scoring within a REST API"
```

Use `^` for PowerShell line continuations (or `\` when running inside bash). You can also store the context in a text file and pass it via `--context-file path/to/context.txt`. The agent automatically calls the Ollama server at `http://127.0.0.1:11434` with the bundled `llama3` model—no additional flags required.

The agent responds with a numbered markdown list explaining 3–5 suitable model classes, reasoning, preprocessing tips, and evaluation considerations that align with the provided metadata and constraints.

## Extending

- Adjust the `DatasetMetadata` schema in `src/model_recommender.py` to encode additional signals (e.g., privacy level, available compute budget).
- Tune the system prompt or temperature to enforce stricter formats or more creative exploration.
- Swap in other LangChain-compatible LLM wrappers if you host a different API; just modify the `ModelRecommenderAgent` constructor accordingly.
