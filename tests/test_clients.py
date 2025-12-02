import pytest

from llm_wrappers.base import BaseLLM


def test_base_generate_raises():
    b = BaseLLM()
    with pytest.raises(NotImplementedError):
        b.generate("hello")


class DummyClient(BaseLLM):
    def generate(self, prompt: str, **kwargs):
        return {"text": f"ECHO: {prompt}", "raw": None}


def test_dummy_client():
    c = DummyClient()
    r = c.generate("hi")
    assert isinstance(r, dict) and r["text"].startswith("ECHO")
