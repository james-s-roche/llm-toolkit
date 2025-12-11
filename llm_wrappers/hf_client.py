import json
import time
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional

import requests


HF_ROUTER_BASE = "https://router.huggingface.co"
HF_HUB_API_BASE = "https://huggingface.co/api"


@dataclass(frozen=True)
class HFModelInfo:
    id: str
    pipeline_tag: Optional[str] = None
    likes: Optional[int] = None
    private: Optional[bool] = None


class HFClient:
    """
    Lightweight Hugging Face client using the router host for inference and the Hub API
    for model discovery.

    - generate(): text generation via POST /models/{model}
    - stream_generate(): best-effort SSE streaming; falls back to one-shot
    - models(): list models from Hub search API (filtering to text-generation by default)

    Auth:
      - Use hf_api_token for both router and hub requests (optional for public models but recommended).
    """

    def __init__(self, model: Optional[str] = None, hf_api_token: Optional[str] = None, timeout: float = 180.0):
        self.model = model
        self.hf_api_token = hf_api_token
        self.timeout = timeout
        self._session = requests.Session()
        if hf_api_token:
            self._session.headers["Authorization"] = f"Bearer {hf_api_token}"
        # Explicitly set UA to help HF debugging if needed
        self._session.headers.setdefault("User-Agent", "llm-wrappers/0.1 (+streamlit)")

    # -------------------------------
    # Models list via HF Hub API
    # -------------------------------
    def models(self, *, pipeline_tag: str = "text-generation", limit: int = 200) -> List[HFModelInfo]:
        params = {"pipeline_tag": pipeline_tag, "limit": str(limit)}
        url = f"{HF_HUB_API_BASE}/models"
        try:
            r = self._session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"Failed to list Hugging Face models: {e}") from e

        infos: List[HFModelInfo] = []
        if isinstance(data, list):
            for m in data:
                mid = str(m.get("modelId") or m.get("id") or "")
                if not mid:
                    continue
                infos.append(
                    HFModelInfo(
                        id=mid,
                        pipeline_tag=m.get("pipeline_tag"),
                        likes=m.get("likes"),
                        private=m.get("private"),
                    )
                )
        return infos

    # -------------------------------
    # Non-streaming generation
    # -------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        extra_parameters: Optional[Dict] = None,
    ) -> str:
        use_model = model or self.model
        if not use_model:
            raise ValueError("No model specified for HF generate()")

        # Payload format accepted by router: inputs + parameters
        text = prompt if not system else f"{system}\n\n{prompt}"
        payload = {
            "inputs": text,
            "parameters": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["parameters"]["max_new_tokens"] = max_tokens
        if extra_parameters:
            payload["parameters"].update(extra_parameters)

        url = f"{HF_ROUTER_BASE}/models/{use_model}"
        try:
            r = self._session.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"HF inference request failed: {e}") from e

        # The response format varies slightly by backend; normalize common shapes
        try:
            data = r.json()
        except Exception:
            # Sometimes TGI streams raw text on non-stream endpoints; treat as plain text
            return r.text or ""

        # Common shapes:
        # - {"generated_text": "..."}
        # - [{"generated_text": "..."}]
        # - {"choices":[{"text":"..."}]}
        if isinstance(data, dict):
            if "generated_text" in data:
                return str(data.get("generated_text") or "")
            choices = data.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                return str(choices[0].get("text") or choices[0].get("generated_text") or "")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return str(data[0].get("generated_text") or data[0].get("text") or "")

        # Fallback to string conversion
        return json.dumps(data)

    # -------------------------------
    # Streaming generation (best effort)
    # -------------------------------
    def stream_generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        extra_parameters: Optional[Dict] = None,
    ) -> Generator[str, None, None]:
        use_model = model or self.model
        if not use_model:
            raise ValueError("No model specified for HF stream_generate()")

        text = prompt if not system else f"{system}\n\n{prompt}"
        parameters = {"temperature": temperature, "stream": True}
        if max_tokens is not None:
            parameters["max_new_tokens"] = max_tokens
        if extra_parameters:
            parameters.update(extra_parameters)

        url = f"{HF_ROUTER_BASE}/models/{use_model}"
        try:
            with self._session.post(url, json={"inputs": text, "parameters": parameters}, timeout=self.timeout, stream=True) as r:
                r.raise_for_status()
                # Try to detect SSE/event-stream; if not, read as text
                ctype = r.headers.get("Content-Type", "")
                if "text/event-stream" in ctype or "application/x-ndjson" in ctype:
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        # SSE lines might be "data: {...}"
                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()
                        try:
                            obj = json.loads(line)
                        except Exception:
                            # Sometimes the router yields plain text chunks
                            yield line
                            continue
                        # TGI stream shape: {"token":{"text":"..."}, "generated_text":null, ...}
                        token = (obj.get("token") or {}).get("text")
                        if token:
                            yield str(token)
                        # Some backends stream choices-like deltas
                        elif "generated_text" in obj and isinstance(obj.get("generated_text"), str):
                            # sometimes final message
                            yield str(obj["generated_text"])
                else:
                    # Not streaming; yield once with parsed text
                    body = r.text or ""
                    try:
                        data = json.loads(body)
                    except Exception:
                        yield body
                    else:
                        if isinstance(data, dict):
                            if "generated_text" in data:
                                yield str(data.get("generated_text") or "")
                            elif isinstance(data.get("choices"), list):
                                ch0 = data["choices"][0]
                                yield str(ch0.get("text") or ch0.get("generated_text") or "")
                            else:
                                yield json.dumps(data)
                        elif isinstance(data, list) and data and isinstance(data[0], dict):
                            yield str(data[0].get("generated_text") or data[0].get("text") or "")
                        else:
                            yield json.dumps(data)
        except Exception as e:
            # Surface the error to the UI
            raise RuntimeError(f"HF streaming request failed: {e}") from e