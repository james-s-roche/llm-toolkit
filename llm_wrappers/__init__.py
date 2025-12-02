"""Top-level llm_wrappers package.

This package exposes the small set of utilities and clients used by the
examples and the Streamlit app. The original project used a `src/` layout; the
modules have been copied here so the repository can be used without an
editable install or PYTHONPATH changes.
"""

from .base import BaseLLM
from .openai_client import OpenAIClient
from .openrouter_client import OpenRouterClient
from .hf_client import HFClient
from .local_client import LocalClient
from .token_counter import TokenCounter, estimate_tokens
from .models_registry import RECOMMENDED
from .secrets import get_secret, set_env_from_secret

__all__ = [
    "BaseLLM",
    "OpenAIClient",
    "OpenRouterClient",
    "HFClient",
    "LocalClient",
    "TokenCounter",
    "estimate_tokens",
    "RECOMMENDED",
    "get_secret",
    "set_env_from_secret",
]
