"""HTTP transport for each supported language-model provider.

Two wire formats cover every provider we support:

  "openai"     chat-completions JSON -- OpenAI, Groq, Gemini (via its
               OpenAI-compatibility endpoint), Hugging Face Inference
               Providers, Cerebras, OpenRouter, Ollama, vLLM, and any other
               OpenAI-compatible server reachable at LLM_BASE_URL.
  "anthropic"  the Anthropic Messages API.

Adding a provider that speaks the OpenAI format needs no code at all -- set
LLM_PROVIDER=custom and LLM_BASE_URL. Only a genuinely different wire format
needs a new function plus a registry entry here.
"""
from typing import Callable, Dict, Optional

import requests

from gridwise import config


class ProviderError(Exception):
    """Transport-level failure: unreachable, unauthorized, rate limited, 5xx.

    Carries `status_code` (None for a connection-level failure with no HTTP
    response at all) and `retry_after` (seconds, from a Retry-After header,
    when the provider sent one) so the retry loop in gridwise.llm.interpreter
    can back off intelligently instead of hammering a rate-limited endpoint.
    """

    # 429 = rate limited; 502/503/504 = gateway/overload errors that are
    # normally transient. Free-tier providers return these far more often
    # than a paid tier would.
    RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

    @property
    def is_retryable(self) -> bool:
        return self.status_code is None or self.status_code in self.RETRYABLE_STATUS_CODES


def _parse_retry_after(response: "requests.Response") -> Optional[float]:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None  # Retry-After may be an HTTP date, which we don't bother parsing.


def _post(url: str, headers: dict, body: dict) -> dict:
    try:
        response = requests.post(
            url, headers=headers, json=body, timeout=config.LLM_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        # Deliberately excludes the response body: a provider error page can
        # echo request headers, and those carry the API key.
        raise ProviderError(f"{type(exc).__name__} contacting provider") from exc

    if response.status_code >= 400:
        raise ProviderError(
            f"provider returned HTTP {response.status_code}",
            status_code=response.status_code,
            retry_after=_parse_retry_after(response),
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError("provider returned a non-JSON body", status_code=response.status_code) from exc


def call_openai_compatible(system_prompt: str, user_prompt: str) -> str:
    """Chat-completions request against any OpenAI-format endpoint."""
    if not config.LLM_BASE_URL:
        raise ProviderError("LLM_BASE_URL is not set for this provider")

    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"

    body = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        # Provider-specific fields (e.g. Gemini's reasoning_effort) -- see
        # PROVIDER_PRESETS in gridwise/config.py. Empty for providers that
        # don't declare any, so their request body is unaffected.
        **config.LLM_EXTRA_PARAMS,
    }

    data = _post(f"{config.LLM_BASE_URL}/chat/completions", headers, body)

    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("unexpected chat-completions response shape") from exc


def call_anthropic(system_prompt: str, user_prompt: str) -> str:
    """Anthropic Messages API request."""
    headers = {
        "x-api-key": config.LLM_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.LLM_MODEL,
        "max_tokens": config.LLM_MAX_TOKENS,
        "temperature": config.LLM_TEMPERATURE,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    data = _post(f"{config.LLM_BASE_URL}/messages", headers, body)

    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise ProviderError("unexpected messages response shape")
    text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    if not text:
        raise ProviderError("provider returned no text content")
    return text


_CALLERS: Dict[str, Callable[[str, str], str]] = {
    "openai": call_openai_compatible,
    "anthropic": call_anthropic,
}


def get_caller(api_style: str) -> Callable[[str, str], str]:
    """Resolves an api_style (from the provider preset) to its transport."""
    caller = _CALLERS.get(api_style)
    if caller is None:
        raise ProviderError(f"unsupported api_style: {api_style}")
    return caller
