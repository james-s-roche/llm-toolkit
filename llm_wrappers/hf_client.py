from typing import Any, Optional

from .base import BaseLLM

try:
    # optional import for remote HF inference
    import requests
except Exception:
    requests = None


class HFClient(BaseLLM):
    def __init__(self, model: str, hf_api_token: Optional[str] = None):
        self.model = model
        self.hf_api_token = hf_api_token

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        if self.hf_api_token and requests:
            url = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {"Authorization": f"Bearer {self.hf_api_token}"}
            resp = requests.post(url, json={"inputs": prompt, **kwargs}, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        else:
            raise RuntimeError("HF remote requires requests and an API token")
