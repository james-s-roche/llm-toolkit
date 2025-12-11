from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import Model


@dataclass(frozen=True)
class OpenAIModelInfo:
    id: str
    created: Optional[int] = None
    owned_by: Optional[str] = None

    @staticmethod
    def from_sdk(m: Model) -> "OpenAIModelInfo":
        # Model objects include id, created, owned_by (and other fields); keep only the basics here.
        return OpenAIModelInfo(
            id=getattr(m, "id", None) or "",
            created=getattr(m, "created", None),
            owned_by=getattr(m, "owned_by", None),
        )


class OpenAIClient:
    """
    Minimal OpenAI client wrapper for openai>=1.x

    - Uses Chat Completions API for text generation.
    - Provides streaming and non-streaming methods.
    - Exposes models() to list available models from the account.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: Optional[float] = 180.0):
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model

    # -------------------------------
    # Models
    # -------------------------------
    def models(self) -> List[OpenAIModelInfo]:
        """Return available models for the account."""
        try:
            resp = self.client.models.list()
            # resp.data is a list[Model]
            return [OpenAIModelInfo.from_sdk(m) for m in resp.data]
        except Exception as e:
            # Let callers decide how to surface this; keep message clear.
            raise RuntimeError(f"Failed to list OpenAI models: {e}") from e

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
    ) -> str:
        """
        Return the assistant text for a single-turn prompt.
        """
        use_model = model or self.model
        if not use_model:
            raise ValueError("No model specified for OpenAI generate()")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            completion: ChatCompletion = self.client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI chat.completions.create failed: {e}") from e

        # The SDK guarantees choices[0].message.content is Optional[str]
        if not completion.choices:
            return ""
        return completion.choices[0].message.content or ""

    # -------------------------------
    # Streaming generation
    # -------------------------------
    def stream_generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Yield assistant text chunks as they arrive. Yields only the delta content strings.
        """
        use_model = model or self.model
        if not use_model:
            raise ValueError("No model specified for OpenAI stream_generate()")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream: Iterable[ChatCompletionChunk] = self.client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI chat.completions.create(stream) failed: {e}") from e

        for chunk in stream:
            # chunk.choices[0].delta.content may be None for non-text deltas; normalize to empty string
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield text