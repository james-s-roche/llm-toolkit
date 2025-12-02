import requests
from typing import Any, Generator, Optional

from .base import BaseLLM


class OllamaClient(BaseLLM):
    """Simple client for a local Ollama server (https://ollama.com).

    By default this will POST to http://localhost:11434/api/generate with JSON
    {"model": <model>, "prompt": <prompt>}. If your Ollama server is on a
    different host/port, pass `host` (e.g. "http://127.0.0.1:11434").
    """

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.endpoint = f"{self.host}/api/generate"

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        payload = {"model": self.model, "prompt": prompt}
        # pass through a few common options if provided
        for k in ("temperature", "max_tokens", "top_p", "top_k"):
            if k in kwargs:
                payload[k] = kwargs[k]

        resp = requests.post(self.endpoint, json=payload, timeout=60)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return resp.text

    def stream_generate(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        # Ollama supports streaming via server-sent events in some setups; for
        # simplicity we fall back to a single response and yield it once.
        out = self.generate(prompt, **kwargs)
        if isinstance(out, dict):
            # attempt to extract text fields
            text = out.get("text") or out.get("output") or out.get("message")
            if text:
                yield str(text)
                return
        yield str(out)
