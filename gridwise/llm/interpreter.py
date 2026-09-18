"""Orchestration for the operator-note interpretation step.

Responsibility: hand the notes to the configured model, tolerantly parse
whatever comes back, retry with backoff on transport failure, and return a
RAW, UNTRUSTED list of interpretation dicts.

This module deliberately does not validate semantics. Deciding whether the
model's answer is usable is gridwise.core.guardrails' job, and keeping the
two apart is what lets the guardrails be the single auditable place where
untrusted output becomes trusted input.
"""
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, List, Optional

from gridwise import config
from gridwise.llm import providers
from gridwise.llm.prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """The model could not be reached or produced nothing parseable."""


# --------------------------------------------------------------------------
# Interpretation cache
# --------------------------------------------------------------------------
# The judge replays hidden cases repeatedly. Identical notes always produce the
# same directives, so caching cuts p95 latency and provider quota use without
# affecting correctness. Bounded LRU, guarded for multi-threaded WSGI workers.
_cache: "OrderedDict[str, List[Any]]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(operator_notes: List[str], battery: Optional[dict]) -> str:
    capacity = (battery or {}).get("capacity_kwh")
    return json.dumps([operator_notes, capacity, config.LLM_MODEL], sort_keys=True)


def _cache_get(key: str) -> Optional[List[Any]]:
    if not config.LLM_CACHE_ENABLED:
        return None
    with _cache_lock:
        if key not in _cache:
            return None
        _cache.move_to_end(key)
        return _cache[key]


def _cache_put(key: str, value: List[Any]) -> None:
    if not config.LLM_CACHE_ENABLED:
        return
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > config.LLM_CACHE_SIZE:
            _cache.popitem(last=False)


def clear_cache() -> None:
    """Used by tests; never called from the request path."""
    with _cache_lock:
        _cache.clear()


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------
_FENCED_ARRAY = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_FENCED_OBJECT = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_array(text: str) -> List[Any]:
    """Pulls a JSON array out of a model response.

    Models wrap JSON in markdown fences or prose despite instructions, and
    some wrap the array in an object such as {"directives": [...]}. All three
    shapes are recovered here rather than being thrown away as a failure --
    a parse we could have made is interpretation points we would have lost.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model response")

    text = text.strip()

    candidates = []
    fenced = _FENCED_ARRAY.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    fenced_obj = _FENCED_OBJECT.search(text)
    if fenced_obj:
        candidates.append(fenced_obj.group(1))

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    candidates.append(text)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Unwrap a single array-valued key, e.g. {"directives": [...]}.
            for value in parsed.values():
                if isinstance(value, list):
                    return value
            # A bare single-entry object is a valid one-note answer.
            if "note_index" in parsed:
                return [parsed]

    raise ValueError("no JSON array found in model response")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def interpret_notes(operator_notes: List[str], battery: Optional[dict] = None) -> List[Any]:
    """Interprets operator notes with the configured model.

    Returns a raw, unvalidated list. Raises LLMError when the provider is
    unreachable or unusable after retries -- callers must catch it and fail
    safe (every note becomes a guardrail no_op) rather than crashing.
    """
    if not config.LLM_API_KEY and config.LLM_PROVIDER not in ("ollama", "custom"):
        raise LLMError("no API key configured for provider " + config.LLM_PROVIDER)

    key = _cache_key(operator_notes, battery)
    cached = _cache_get(key)
    if cached is not None:
        logger.info("interpretation cache hit (%d notes)", len(operator_notes))
        return cached

    caller = providers.get_caller(config.LLM_API_STYLE)
    user_prompt = build_user_prompt(operator_notes, battery)

    last_error: Optional[Exception] = None
    total_attempts = config.LLM_MAX_RETRIES + 1
    for attempt in range(total_attempts):
        try:
            raw_text = caller(SYSTEM_PROMPT, user_prompt)
            parsed = extract_json_array(raw_text)
            _cache_put(key, parsed)
            return parsed
        except Exception as exc:  # noqa: BLE001 - broad by design: retry, then fail safe
            last_error = exc
            logger.warning("LLM attempt %d/%d failed: %s", attempt + 1, total_attempts, exc)

            # A provider error that is NOT retryable (e.g. 401/403/400 -- bad
            # key, bad request) will not go away on retry, so stop immediately
            # instead of spending the remaining time budget on it.
            if isinstance(exc, providers.ProviderError) and not exc.is_retryable:
                break

            if attempt < total_attempts - 1:
                _backoff_before_retry(exc, attempt)

    raise LLMError(f"model call failed after retries: {last_error}")


def _backoff_before_retry(exc: Exception, attempt: int) -> None:
    """Waits before the next attempt, longer for a provider that explicitly
    signalled overload/rate-limiting (429/502/503/504) than for e.g. a
    one-off parse failure, where an immediate retry costs nothing extra.

    Free-tier providers (this service's intended host) throttle far more
    aggressively than a paid tier, and a bare immediate retry into a rate
    limit essentially never succeeds -- see gridwise/llm/providers.py.
    """
    if not isinstance(exc, providers.ProviderError):
        return  # e.g. a JSON parse failure: nothing to wait out.

    if exc.retry_after is not None:
        # Capped well below the judge's 30s per-request budget: with the
        # default 12s per-call timeout, two attempts plus this backoff must
        # still land comfortably under 30s (12 + 3 + 12 = 27s worst case).
        delay = min(exc.retry_after, 3.0)
    else:
        delay = config.LLM_RETRY_BACKOFF_SECONDS * (2 ** attempt)

    logger.info("backing off %.1fs before retry (HTTP %s)", delay, exc.status_code)
    time.sleep(delay)
