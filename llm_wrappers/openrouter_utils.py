import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

CACHE_PATH = os.path.join(os.getcwd(), ".openrouter_models_cache.json")
CACHE_TTL = 60 * 60 * 6  # 6 hours freshness (but we fetch on each launch by default)


def fetch_openrouter_models(api_url: str = "https://openrouter.ai/api/v1/models") -> List[Dict[str, Any]]:
    """Fetch models from OpenRouter API. Returns a list of model dicts.

    On network failure, falls back to a cached copy in the repo root if available.
    """
    try:
        resp = requests.get(api_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data if isinstance(data, list) else data.get("models") or data.get("data") or []
        # persist cache
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"fetched_at": time.time(), "models": models}, f)
        except Exception:
            pass
        return models
    except Exception:
        # fallback to cache
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return cached.get("models", [])
        except Exception:
            return []


def is_free_model(model: Dict[str, Any]) -> bool:
    """Heuristic: model id endswith ':free' or tags / pricing indicate free."""
    mid = str(model.get("id") or model.get("model") or "")
    if mid.endswith(":free"):
        return True
    # try fields
    tags = model.get("tags") or model.get("label") or []
    if isinstance(tags, list) and any("free" in str(t).lower() for t in tags):
        return True
    price = model.get("price") or model.get("pricing")
    if price in (None, 0, "free", "Free"):
        return True
    return False


def sort_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort models so free models (id ending with :free) come first, then by id."""
    def key(m: Dict[str, Any]):
        mid = str(m.get("id") or m.get("model") or "")
        free = 0 if mid.endswith(":free") or is_free_model(m) else 1
        return (free, mid.lower())

    return sorted(models, key=key)


def provider_of(model: Dict[str, Any]) -> Optional[str]:
    """Heuristic to extract a provider name for display and filtering.

    Priority:
    1. model['provider'] | model['vendor'] | model['source'] (if present)
       - if a dict, prefer the 'name' field
    2. Derive from the model id/model string by taking the text before the first '/'
       (e.g. 'alibaba/tongyi-...:free' -> 'alibaba'). Also strip trailing
       tags like ':free' before extracting.
    3. Fallback to a cleaned, lower-cased form of the id/model string.
    Returns None when no reasonable provider can be derived.
    """
    p = model.get("provider") or model.get("vendor") or model.get("source")
    # provider can be a dict or string
    if isinstance(p, dict):
        name = p.get("name")
        return name.lower() if name else None
    if p:
        return str(p).lower()

    # derive from id/model/name fields
    mid = str(model.get("id") or model.get("model") or model.get("name") or "").strip()
    if not mid:
        return None

    # strip any trailing tag like ':free' or ':paid'
    if ":" in mid:
        mid = mid.split(":", 1)[0]

    # if id contains provider/model (provider/model-name), take provider
    if "/" in mid:
        return mid.split("/", 1)[0].lower()

    # as a best-effort fallback, return the start of the id (split on hyphen)
    if "-" in mid:
        return mid.split("-", 1)[0].lower()

    return mid.lower() or None
