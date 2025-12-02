from typing import Any


class BaseLLM:
    """Simple base class for LLM clients.

    Contract:
      - generate(prompt: str, **kwargs) -> dict | str
        Accepts a prompt (string) and returns either a string or a dict with more details.
    """

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate text from `prompt`.

        Implementations should return either a string (the generated text) or a dict
        containing more detailed response metadata.
        """
        raise NotImplementedError("Subclasses must implement generate")

    def stream_generate(self, prompt: str, **kwargs: Any):
        """Streaming generator interface.

        Default implementation calls `generate` and yields the full text once.
        Subclasses can override to provide streaming tokens.
        """
        out = self.generate(prompt, **kwargs)
        if isinstance(out, dict) and "text" in out:
            yield out["text"]
        else:
            yield str(out)
