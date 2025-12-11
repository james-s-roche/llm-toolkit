import os
import math
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from llm_wrappers.models_registry import RECOMMENDED
from llm_wrappers.token_counter import estimate_tokens
from llm_wrappers.openai_client import OpenAIClient, OpenAIModelInfo
from llm_wrappers.hf_client import HFClient, HFModelInfo
from llm_wrappers.local_client import LocalClient
from llm_wrappers.ollama_client import OllamaClient
from llm_wrappers.openrouter_client import OpenRouterClient
from llm_wrappers.secrets import get_secret
from llm_wrappers.openrouter_utils import (
    fetch_openrouter_models,
    sort_models,
    is_free_model,
    provider_of,
)

# Expose for tabs
BACKENDS = ["openrouter", "openai", "huggingface", "local"]
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cache OpenRouter models at import time
OPENROUTER_MODELS: List[Dict[str, Any]] = []
try:
    OPENROUTER_MODELS = sort_models(fetch_openrouter_models())
except Exception:
    OPENROUTER_MODELS = []

def ensure_session():
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = []
    if "templates" not in st.session_state:
        st.session_state["templates"] = {
            "Greeting": "Write a friendly greeting in one sentence.",
            "ELI5": "Explain {topic} like I'm 5.",
            "Summarize": "Summarize the following text:\n{input}",
        }

def add_message(entry: Dict[str, Any]):
    st.session_state["conversations"].append(entry)

def render_conversation():
    for msg in reversed(st.session_state.get("conversations", [])):
        with st.expander(f"[{msg.get('backend','')}/{msg.get('model','')}] {msg.get('role','resp')}"):
            st.markdown("**Prompt**")
            st.write(msg.get("prompt", ""))
            st.markdown("**Response**")
            st.write(msg.get("response", ""))
            st.markdown("**Meta**")
            st.write({k: v for k, v in msg.items() if k not in ("prompt", "response")})

def _extract_ollama_model_name(model: str) -> Tuple[str, Optional[str]]:
    if isinstance(model, str) and (model.startswith("ollama:") or model.startswith("ollama/")):
        cleaned = model.split(":", 1)[1] if ":" in model else model.split("/", 1)[1]
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        return cleaned, host
    return model, None

def get_client(backend: str, model: str):
    backend = backend.lower()
    if backend == "openai":
        api_key = get_secret("OPENAI_API_KEY")
        return OpenAIClient(model=model, api_key=api_key)
    if backend == "huggingface":
        hf_token = get_secret("HF_API_TOKEN")
        return HFClient(model=model, hf_api_token=hf_token)
    if backend == "local":
        cleaned, host = _extract_ollama_model_name(model)
        if host:
            return OllamaClient(model=cleaned, host=host)
        return LocalClient(model=cleaned)
    if backend == "openrouter":
        api_key = get_secret("OPENROUTER_API_KEY")
        return OpenRouterClient(api_url=OPENROUTER_API_URL, api_key=api_key, model=model)
    raise ValueError(f"Unknown backend: {backend}")

def get_recommended_models(backend: str) -> List[str]:
    b = backend.lower()
    if b == "openai":
        return [m["id"] for m in RECOMMENDED.get("openai", [])]
    if b == "huggingface":
        return [m["id"] for m in RECOMMENDED.get("huggingface", [])]
    if b == "openrouter":
        if OPENROUTER_MODELS:
            return [str(m.get("id") or m.get("model") or m.get("name") or "") for m in OPENROUTER_MODELS]
        return [m["id"] for m in RECOMMENDED.get("openrouter", [])]
    return [m["id"] for m in RECOMMENDED.get("local", [])]

def _openrouter_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    for m in OPENROUTER_MODELS:
        mid = str(m.get("id") or m.get("model") or "")
        if mid == model_id:
            return m
    return None

def _get_max_tokens_from_openrouter_model(m: Dict[str, Any]) -> Optional[int]:
    for k in ("max_tokens", "context", "context_window", "max_context", "max_input_tokens", "input_context", "context_length"):
        v = m.get(k)
        try:
            if isinstance(v, (int, float)) and int(v) > 0:
                return int(v)
            if isinstance(v, str) and v.isdigit():
                return int(v)
        except Exception:
            continue
    return None

def token_slider_options(backend: str, model: str) -> List[int]:
    token_max = 4096
    if backend.lower() == "openrouter":
        meta = _openrouter_model_info(model)
        if meta:
            token_max = _get_max_tokens_from_openrouter_model(meta) or token_max
    step_max = max(4, int(math.sqrt(token_max)))
    return [2**n for n in range(4, min(step_max, 14) + 1)]

def filtered_openrouter_model_ids(free_only: bool, provider_sel: str) -> List[str]:
    if not OPENROUTER_MODELS:
        return [m["id"] for m in RECOMMENDED.get("openrouter", [])]
    ids: List[str] = []
    for m in OPENROUTER_MODELS:
        if free_only and not is_free_model(m):
            continue
        prov = provider_of(m) or "unknown"
        if provider_sel != "All" and prov != provider_sel:
            continue
        mid = str(m.get("id") or m.get("model") or m.get("name") or "")
        if mid:
            ids.append(mid)
    return ids

def credentials_status() -> Dict[str, bool]:
    return {
        "OPENAI_API_KEY": bool(get_secret("OPENAI_API_KEY")),
        "HF_API_TOKEN": bool(get_secret("HF_API_TOKEN")),
        "OPENROUTER_API_KEY": bool(get_secret("OPENROUTER_API_KEY")),
    }

def estimate(s: str) -> int:
    return estimate_tokens(s or "")

def refresh_openrouter_models() -> int:
    try:
        new = sort_models(fetch_openrouter_models())
        import sys
        mod = sys.modules[__name__]
        setattr(mod, "OPENROUTER_MODELS", new)
        return len(new)
    except Exception:
        return 0

# -------------------------------
# OpenAI models lightweight cache
# -------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def list_openai_models_cached(api_key: Optional[str]) -> List[OpenAIModelInfo]:
    if not api_key:
        return []
    client = OpenAIClient(api_key=api_key)
    try:
        return client.models()
    except Exception:
        return []

def get_openai_model_ids() -> List[str]:
    api_key = get_secret("OPENAI_API_KEY")
    live = list_openai_models_cached(api_key)
    if live:
        ids = [m.id for m in live if m.id]
        priority = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo"]
        ids = sorted(set(ids), key=lambda x: (0 if x in priority else 1, x))
        return ids
    return [m["id"] for m in RECOMMENDED.get("openai", [])]

# -------------------------------
# Hugging Face models lightweight cache
# -------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def list_hf_models_cached(hf_token: Optional[str], limit: int = 300) -> List[HFModelInfo]:
    client = HFClient(hf_api_token=hf_token)
    try:
        return client.models(pipeline_tag="text-generation", limit=limit)
    except Exception:
        return []

def get_hf_model_ids() -> List[str]:
    hf_token = get_secret("HF_API_TOKEN")
    live = list_hf_models_cached(hf_token)
    if live:
        # Prefer popular chatty models near top
        priority = ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1", "HuggingFaceH4/zephyr-7b-beta"]
        ids = [m.id for m in live]
        ids = sorted(set(ids), key=lambda x: (0 if x in priority else 1, x))
        return ids
    return [m["id"] for m in RECOMMENDED.get("huggingface", [])]