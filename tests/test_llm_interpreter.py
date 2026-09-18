"""Unit tests for gridwise.llm.interpreter -- parsing and caching only.

No real network calls are made: gridwise.llm.providers.get_caller is
monkeypatched so these tests run offline and do not need an API key.
"""
import json

import pytest

from gridwise import config
from gridwise.llm import interpreter


@pytest.fixture(autouse=True)
def _clear_cache():
    interpreter.clear_cache()
    yield
    interpreter.clear_cache()


def test_extract_json_array_plain():
    text = '[{"note_index": 0, "applies": false}]'
    assert interpreter.extract_json_array(text) == [{"note_index": 0, "applies": False}]


def test_extract_json_array_with_markdown_fence():
    text = '```json\n[{"note_index": 0, "applies": false}]\n```'
    assert interpreter.extract_json_array(text) == [{"note_index": 0, "applies": False}]


def test_extract_json_array_with_surrounding_prose():
    text = 'Here is the result:\n[{"note_index": 0, "applies": false}]\nDone.'
    assert interpreter.extract_json_array(text) == [{"note_index": 0, "applies": False}]


def test_extract_json_array_wrapped_in_object():
    text = '{"directives": [{"note_index": 0, "applies": false}]}'
    assert interpreter.extract_json_array(text) == [{"note_index": 0, "applies": False}]


def test_extract_json_array_single_object_for_one_note():
    text = '{"note_index": 0, "applies": false, "directive_type": "no_op"}'
    result = interpreter.extract_json_array(text)
    assert result == [{"note_index": 0, "applies": False, "directive_type": "no_op"}]


def test_extract_json_array_raises_on_garbage():
    with pytest.raises(ValueError):
        interpreter.extract_json_array("not json at all")


def test_extract_json_array_raises_on_empty():
    with pytest.raises(ValueError):
        interpreter.extract_json_array("")


def test_interpret_notes_uses_configured_caller(monkeypatch):
    calls = []

    def fake_caller(system_prompt, user_prompt):
        calls.append((system_prompt, user_prompt))
        return json.dumps([
            {"note_index": 0, "applies": False, "directive_type": "no_op",
             "structured_adjustment": None, "explanation": "distractor"}
        ])

    monkeypatch.setattr(config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(interpreter.providers, "get_caller", lambda style: fake_caller)

    result = interpreter.interpret_notes(["The cafeteria menu changes tomorrow."])
    assert result[0]["directive_type"] == "no_op"
    assert len(calls) == 1


def test_interpret_notes_caches_identical_requests(monkeypatch):
    call_count = {"n": 0}

    def fake_caller(system_prompt, user_prompt):
        call_count["n"] += 1
        return json.dumps([
            {"note_index": 0, "applies": False, "directive_type": "no_op",
             "structured_adjustment": None, "explanation": "cached?"}
        ])

    monkeypatch.setattr(config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", True)
    monkeypatch.setattr(interpreter.providers, "get_caller", lambda style: fake_caller)

    notes = ["Same note text."]
    interpreter.interpret_notes(notes)
    interpreter.interpret_notes(notes)
    assert call_count["n"] == 1, "second identical call should hit the cache"


def test_interpret_notes_raises_llm_error_without_api_key(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    with pytest.raises(interpreter.LLMError):
        interpreter.interpret_notes(["some note"])


def test_interpret_notes_retries_then_raises_on_persistent_failure(monkeypatch):
    def failing_caller(system_prompt, user_prompt):
        raise RuntimeError("provider down")

    monkeypatch.setattr(config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 1)
    monkeypatch.setattr(interpreter.providers, "get_caller", lambda style: failing_caller)

    with pytest.raises(interpreter.LLMError):
        interpreter.interpret_notes(["some note"])


def test_interpret_notes_backs_off_on_retryable_provider_error(monkeypatch):
    from gridwise.llm.providers import ProviderError

    call_count = {"n": 0}
    sleeps = []

    def rate_limited_then_ok(system_prompt, user_prompt):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ProviderError("rate limited", status_code=429, retry_after=0.01)
        return json.dumps([
            {"note_index": 0, "applies": False, "directive_type": "no_op",
             "structured_adjustment": None, "explanation": "ok on retry"}
        ])

    monkeypatch.setattr(config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 1)
    monkeypatch.setattr(interpreter.providers, "get_caller", lambda style: rate_limited_then_ok)
    monkeypatch.setattr(interpreter.time, "sleep", lambda s: sleeps.append(s))

    result = interpreter.interpret_notes(["some note"])
    assert result[0]["explanation"] == "ok on retry"
    assert call_count["n"] == 2
    assert sleeps == [0.01]  # honored the small Retry-After instead of the default backoff


def test_interpret_notes_stops_immediately_on_non_retryable_error(monkeypatch):
    from gridwise.llm.providers import ProviderError

    call_count = {"n": 0}

    def unauthorized(system_prompt, user_prompt):
        call_count["n"] += 1
        raise ProviderError("bad key", status_code=401)

    monkeypatch.setattr(config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)  # would allow 3 attempts if it retried
    monkeypatch.setattr(interpreter.providers, "get_caller", lambda style: unauthorized)

    with pytest.raises(interpreter.LLMError):
        interpreter.interpret_notes(["some note"])

    assert call_count["n"] == 1, "a 401 should not be retried at all"
