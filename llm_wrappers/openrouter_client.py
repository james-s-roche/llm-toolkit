import requests
from typing import Any, Optional

from .base import BaseLLM


class OpenRouterClient(BaseLLM):
    def __init__(self, endpoint: str = None, api_url: str = None, api_key: str = None, model: str = None):
        # accept either `endpoint` or `api_url` keyword to be compatible with callers
        url = api_url or endpoint
        if not url:
            raise ValueError("OpenRouterClient requires `api_url` or `endpoint`")
        self.endpoint = url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        # Build payload depending on endpoint type. OpenRouter's chat/completions
        # endpoint expects a `messages` list; other completions endpoints may
        # accept `input`.
        payload: dict = {}
        if self.model:
            payload["model"] = self.model

        # If endpoint looks like a chat/completions endpoint, send `messages`.
        if "chat" in self.endpoint or "chat/completions" in self.endpoint:
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["input"] = prompt

        # allow pass-through of temperature, max_tokens, etc.
        payload.update(kwargs)

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)

        # Provide a clearer error message on 4xx/5xx responses including body
        if resp.status_code >= 400:
            text = resp.text
            raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {text}", response=resp)

        try:
            data = resp.json()
        except Exception:
            return resp.text

        # Extract textual output from common shapes
        return self._extract_text_from_response(data)

    def _extract_text_from_response(self, data: Any) -> Optional[str]:
        # Try several common response shapes to find the generated text
        if data is None:
            return None
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            # OpenAI-like chat: choices[0].message.content
            choices = data.get("choices") or data.get("outputs") or data.get("data")
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                # check nested message
                if isinstance(first, dict):
                    # common: {'message': {'content': '...'}}
                    msg = first.get("message") or first.get("output") or first.get("text")
                    if isinstance(msg, dict) and "content" in msg:
                        return msg.get("content")
                    if isinstance(msg, str):
                        return msg
                    # sometimes choice has 'text' directly
                    if "text" in first:
                        return first.get("text")
                    # sometimes choice has 'message' with 'content' at deeper level
                    if "message" in first and isinstance(first.get("message"), dict):
                        return first["message"].get("content")

            # direct fields
            for k in ("text", "output", "message", "result"):
                if k in data:
                    v = data.get(k)
                    if isinstance(v, str):
                        return v
                    if isinstance(v, dict) and "content" in v:
                        return v.get("content")

        # fallback: pretty-print the JSON as a string
        try:
            return str(data)
        except Exception:
            return None

    def stream_generate(self, prompt: str, **kwargs: Any):
        # Default streaming: call generate and yield the full text once. This
        # can be extended to support SSE streaming if desired.
        out = self.generate(prompt, **kwargs)
        if isinstance(out, dict) and "text" in out:
            yield out["text"]
        else:
            yield str(out)
