# llm-wrappers

Lightweight Python wrappers to experiment with several LLM backends (OpenRouter, OpenAI, Hugging Face Inference API, and local Transformers). 

## What's included

- `llm_wrappers/` — the small Python package with the base client and concrete clients.
- `examples/interactive_cli.py` — simple CLI for manual testing and quick experiments.
- `webui/streamlit_app.py` — Streamlit-based playground and benchmark dashboard.
- `scripts/benchmark.py` — lightweight benchmarking harness for latency/token measurements.
- `tests/` — unit tests (run with `pytest`).
- `requirements.txt` — recommended optional dependencies.

## Quickstart (no install required)

1. (Optional) create and activate a virtual environment:

```
python -m venv llms_env
source source llms_env/bin/activate
```

2. Install the optional dependencies you need. For example, to run the Streamlit UI and tests:

```
pip install -r requirements.txt
```

3. Run tests (the package is importable directly from the repository root):

```
PYTHONPATH=. pytest -q
```

4. Run the interactive CLI (it will use environment variables, `.env`, or macOS Keychain for API keys):

```
python examples/interactive_cli.py --backend openai --model gpt-3.5-turbo
```

5. Run the Streamlit UI (recommended):

```bash
streamlit run webui/streamlit_app.py
```

Because the package is top-level (`llm_wrappers/`) you do not need to set `PYTHONPATH` or install the package to import it from the examples and the web UI.

## Environment variables / secrets

Set API keys as environment variables or store them in a `.env` file in the repository root (development only). The included `llm_wrappers.secrets` helper will also try the macOS Keychain.

- `OPENAI_API_KEY` — OpenAI API key
- `HF_API_TOKEN` — Hugging Face Inference API token

Example `.env` (do not commit):

```text
OPENAI_API_KEY=sk-...yourkey...
HF_API_TOKEN=hf_...yourtoken...
```

Or add to macOS Keychain:

```bash
security add-generic-password -s OPENAI_API_KEY -w "sk-...yourkey..." -U
```

## Notes

- The repo is intentionally minimal: no editable install or `src/` layout is required anymore.
- Packaging/config files (`pyproject.toml` and `setup.cfg`) were removed to keep the tree simple. If you want packaging back (for pip install), I can re-add them.
- If you prefer a one-line launcher for Streamlit that sets environment variables or activates a virtualenv then runs Streamlit, say the word and I'll add it.

If you'd like, I can also remove the now-empty `src/` directory entirely or update the Makefile to remove `PYTHONPATH` helpers — tell me which cleanup items you prefer.

## Local vs Hosted models (quick notes)

- Local models (selected via the "local" backend) use the Hugging Face `transformers` library by default and will download model weights from the Hugging Face Hub the first time you use them. This is not a standalone GUI app, but it will download files (tens to hundreds of MB for small models, multiple GB for larger ones) and requires CPU/RAM to run inference. The default local model in this project is `distilgpt2` (a small distilled GPT-2 variant) which is lighter-weight than `gpt2`.

- Hosted models (OpenRouter, Hugging Face Inference API, etc.) run remotely — you send the prompt and receive responses from a provider. Hosted models avoid local weight downloads and heavy CPU/memory usage, but they require network access and may need API keys, have rate limits, or incur usage costs.

- Ollama is an option for running models locally behind a small local server, but it requires installing the Ollama app. If you want to avoid installing apps, prefer the lightweight Hugging Face models (e.g. `distilgpt2`, `EleutherAI/gpt-neo-125M`) which download weights via `transformers` and run directly in Python.

- Disk + performance notes:
	- Small models (distilled or 100–300M parameters) typically download in the tens to hundreds of MB and can run reasonably on a modern laptop CPU.
	- Medium/large models (1B+ parameters) may be several GB and are slow on CPU — consider hosted inference or a machine with a GPU.
	- Check your Hugging Face cache (usually `~/.cache/huggingface/transformers`) for downloaded model files.

If you'd like, I can add a UI hint that explains the difference between local/hosted models and shows approximate download sizes for the example models.
