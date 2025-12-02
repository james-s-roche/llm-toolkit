from typing import Optional

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from transformers import GPT2TokenizerFast
except Exception:
    GPT2TokenizerFast = None


class TokenCounter:
    def __init__(self, model: Optional[str] = None):
        self.model = model
        self._enc = None
        if tiktoken is not None:
            try:
                # try to select a tiktoken encoding
                self._enc = tiktoken.get_encoding("gpt2")
            except Exception:
                self._enc = None
        if self._enc is None and GPT2TokenizerFast is not None:
            self._enc = GPT2TokenizerFast.from_pretrained("gpt2")

    def count(self, text: str) -> int:
        if self._enc is None:
            # simple fallback: whitespace count
            return max(1, len(text.split()))
        if hasattr(self._enc, "encode"):
            return len(self._enc.encode(text))
        # huggingface tokenizer uses encode_plus or __call__
        tokens = self._enc(text)
        if isinstance(tokens, dict) and "input_ids" in tokens:
            return len(tokens["input_ids"])
        if isinstance(tokens, (list, tuple)):
            return len(tokens)
        return len(str(tokens))


def estimate_tokens(text: str) -> int:
    return TokenCounter().count(text)
