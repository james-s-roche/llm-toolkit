from typing import Any, Generator, Optional
import json

from .base import BaseLLM


class OpenAIClient(BaseLLM):
    """OpenAI client for openai>=1.0.0 (the new OpenAI Python SDK).

    This client constructs an `openai.OpenAI` client instance and uses
    `client.chat.completions.create` and `client.chat.completions.stream`.

    It intentionally drops support for the pre-1.0 `openai.ChatCompletion` API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        try:
            # import on demand so the package can be imported without optional deps
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai>=1.0 is required for OpenAIClient; install with `pip install openai`") from e

        # Instantiate the modern OpenAI client. If api_key is provided we pass it
        # via the environment variable which the SDK will pick up, otherwise rely
        # on standard configuration.
        if api_key:
            # set temporary attribute on env for client (the SDK also accepts explicit config in some versions)
            import os

            os.environ.setdefault("OPENAI_API_KEY", api_key)

        try:
            self._client = OpenAI()
        except Exception as e:
            raise RuntimeError("Failed to initialize OpenAI client: ensure openai package is configured and OPENAI_API_KEY is set") from e

        self.model = model

    def _extract_text_from_choice(self, choice: Any) -> Optional[str]:
        # Handle multiple response shapes: object attrs or dicts.
        try:
            # object style: choice.message.content
            msg = getattr(choice, "message", None)
            if msg is not None:
                cont = getattr(msg, "content", None)
                if cont:
                    return cont
        except Exception:
            pass

        try:
            # dict style
            if isinstance(choice, dict):
                msg = choice.get("message")
                if isinstance(msg, dict):
                    return msg.get("content")
        except Exception:
            pass

        # fallback None
        return None

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # Use the modern client create API
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **kwargs)

        # Try to extract the primary text from the response
        try:
            # object-like
            first = resp.choices[0]
            text = self._extract_text_from_choice(first)
            if text:
                return text
        except Exception:
            pass

        try:
            # dict-like
            if isinstance(resp, dict):
                return resp.get("choices", [])[0].get("message", {}).get("content")
        except Exception:
            pass

        # Last resort: return string representation
        return str(resp)

    def stream_generate(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # The modern SDK exposes a `stream` helper which yields events. We'll
        # iterate and extract token deltas when possible. If streaming isn't
        # available, fall back to a single generate call.
        try:
            stream = self._client.chat.completions.stream(model=self.model, messages=messages, **kwargs)
        except Exception:
            # streaming may not be supported in this environment; fall back
            yield self.generate(prompt, **kwargs)
            return

        # Iterate the stream and yield textual chunks
        for event in stream:
            try:
                # event may have choices with delta
                choices = getattr(event, "choices", None) or (event.get("choices") if isinstance(event, dict) else None)
                if choices:
                    for c in choices:
                        # delta can be object or dict
                        delta = getattr(c, "delta", None) or (c.get("delta") if isinstance(c, dict) else None)
                        if delta is None:
                            # some events embed a full message
                            text = self._extract_text_from_choice(c)
                            if text:
                                yield text
                        else:
                            # delta may contain 'content'
                            if isinstance(delta, dict):
                                text = delta.get("content")
                            else:
                                text = getattr(delta, "content", None)
                            if text:
                                yield text
                else:
                    # fallback: stringify event
                    try:
                        yield json.dumps(event)
                    except Exception:
                        yield str(event)
            except GeneratorExit:
                raise
            except Exception:
                # ignore malformed stream items
                continue
import json
from typing import Any, Generator, Optional

from .base import BaseLLM


class OpenAIClient(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", openai_module: Optional[Any] = None):
        """Initialize the OpenAI client.

        Accepts either an `openai` module instance via `openai_module` (useful for injection/testing),
        or an `api_key` which will be used to configure the real `openai` package when available.
        """
        self.model = model
        if openai_module is not None:
            self.openai = openai_module
        else:
            try:
                import openai

                # some versions require constructing a client object
                if api_key and hasattr(openai, "api_key"):
                    openai.api_key = api_key
                self.openai = openai
            except Exception as e:  # pragma: no cover - optional dependency
                raise RuntimeError("The 'openai' package is required for OpenAIClient. Install with `pip install openai`") from e

        # Detect whether the installed openai module uses the old ChatCompletion API
        # (pre-1.0) or the new OpenAI client (post-1.0). Prefer a version check when
        # available because some newer installs still expose compatibility symbols
        # that raise at runtime.
        ver = getattr(self.openai, "__version__", None)
        major = 0
        if ver:
            try:
                major = int(str(ver).split(".")[0])
            except Exception:
                major = 0

        self._old_api = hasattr(self.openai, "ChatCompletion") and major < 1
        self._client = None
        if not self._old_api and hasattr(self.openai, "OpenAI"):
            try:
                # instantiate the newer client (this may raise if credentials missing)
                self._client = self.openai.OpenAI()
            except Exception:
                # fall back to None; we'll attempt to call module-level functions if present
                self._client = None

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # Call using the detected API surface.
        if self._old_api:
            resp = self.openai.ChatCompletion.create(model=self.model, messages=messages, **kwargs)
        else:
            # try new client if available
            if self._client is not None:
                resp = self._client.chat.completions.create(model=self.model, messages=messages, **kwargs)
            else:
                # last resort: try module-level helper (some installs expose a `chat` attribute)
                try:
                    resp = self.openai.ChatCompletion.create(model=self.model, messages=messages, **kwargs)
                except Exception:
                    raise RuntimeError("OpenAI client not configured correctly; install openai and/or provide api_key")

        # Normalize result extraction across possible shapes
        try:
            return resp.choices[0].message.content
        except Exception:
            try:
                return resp["choices"][0]["message"]["content"]
            except Exception:
                try:
                    # new API may return nested dicts
                    first = getattr(resp, "data", None) or resp
                    if isinstance(first, dict):
                        return first.get("choices", [{}])[0].get("message", {}).get("content") or str(first)
                except Exception:
                    pass
                return str(resp)

    def stream_generate(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # Try streaming on the API surface we detected. If streaming is unsupported,
        # fall back to a single generate call and yield the whole text.
        try:
            if self._old_api:
                iterator = self.openai.ChatCompletion.create(model=self.model, messages=messages, stream=True, **kwargs)
                for chunk in iterator:
                    if hasattr(chunk, "choices"):
                        for c in chunk.choices:
                            delta = getattr(c, "delta", None)
                            if delta and isinstance(delta, dict):
                                text = delta.get("content")
                                if text:
                                    yield text
                    else:
                        data = getattr(chunk, "data", None) or chunk
                        try:
                            s = json.dumps(data)
                        except Exception:
                            s = str(data)
                        yield s
            else:
                # New API: try streaming via the client if available
                if self._client is not None:
                    # many installs support an iterator-style stream() call
                    stream_iter = None
                    try:
                        stream_iter = self._client.chat.completions.stream(model=self.model, messages=messages, **kwargs)
                    except Exception:
                        # not available; try create with stream=True
                        try:
                            stream_iter = self._client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs)
                        except Exception:
                            stream_iter = None

                    if stream_iter is not None:
                        for chunk in stream_iter:
                            try:
                                # chunk may be an object with delta
                                if hasattr(chunk, "choices"):
                                    for c in chunk.choices:
                                        delta = getattr(c, "delta", None)
                                        if delta and isinstance(delta, dict):
                                            text = delta.get("content")
                                            if text:
                                                yield text
                                else:
                                    data = getattr(chunk, "data", None) or chunk
                                    try:
                                        s = json.dumps(data)
                                    except Exception:
                                        s = str(data)
                                    yield s
                            except Exception:
                                continue
                    else:
                        # streaming not available; fall back
                        full = self.generate(prompt, **kwargs)
                        yield full
                else:
                    # no client available; fall back
                    full = self.generate(prompt, **kwargs)
                    yield full
        except Exception:
            full = self.generate(prompt, **kwargs)
            yield full
