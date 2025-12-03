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
