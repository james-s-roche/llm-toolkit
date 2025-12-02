RECOMMENDED = {
    # For web UI we use a list of dicts with an 'id' key so callers can
    # display richer metadata later if desired.
    "openai": [
        {"id": "gpt-4o-mini"},
        {"id": "gpt-4o"},
        {"id": "gpt-3.5-turbo"},
    ],
    "huggingface": [
        {"id": "bigscience/bloom-1b1"},
        {"id": "mistral-large"},
    ],
    "openrouter": [
        {"id": "gemma-1.0"},
        {"id": "mistral-large"},
    ],
    "local": [
        {"id": "gpt2"},
    ],
}
