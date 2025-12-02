from typing import Any, Optional

from .base import BaseLLM

try:
    from transformers import pipeline
except Exception:
    pipeline = None


class LocalClient(BaseLLM):
    def __init__(self, model: str = "distilgpt2", device: Optional[int] = None):
        if pipeline is None:
            raise RuntimeError("transformers not available")
        self.model = model
        self.device = device
        self._pipe = pipeline("text-generation", model=model, device=device)

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        out = self._pipe(prompt, max_length=kwargs.get("max_length", 200), do_sample=kwargs.get("do_sample", True))
        # pipeline returns list of generations
        if isinstance(out, list) and len(out) > 0:
            return out[0].get("generated_text", out[0])
        return out
