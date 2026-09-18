"""Unit tests for gridwise.llm.providers -- ProviderError classification and
extra_params passthrough. No real network calls (requests.post is mocked).
"""
from unittest.mock import MagicMock, patch

from gridwise import config
from gridwise.llm import providers


def test_provider_error_retryable_for_429_and_5xx():
    for code in (429, 502, 503, 504):
        assert providers.ProviderError("x", status_code=code).is_retryable is True


def test_provider_error_not_retryable_for_4xx_auth_errors():
    for code in (400, 401, 403, 404):
        assert providers.ProviderError("x", status_code=code).is_retryable is False


def test_provider_error_retryable_when_status_unknown():
    # A connection-level failure (no HTTP response at all) is retried too.
    assert providers.ProviderError("x", status_code=None).is_retryable is True


def _mock_response(status_code=200, json_body=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = json_body or {}
    return resp


def test_post_extracts_retry_after_header_on_429():
    resp = _mock_response(status_code=429, headers={"Retry-After": "2.5"})
    with patch("gridwise.llm.providers.requests.post", return_value=resp):
        try:
            providers._post("http://example.test", {}, {})
            assert False, "expected ProviderError"
        except providers.ProviderError as exc:
            assert exc.status_code == 429
            assert exc.retry_after == 2.5
            assert exc.is_retryable is True


def test_post_handles_non_numeric_retry_after_gracefully():
    resp = _mock_response(status_code=503, headers={"Retry-After": "Wed, 21 Oct too-far-away"})
    with patch("gridwise.llm.providers.requests.post", return_value=resp):
        try:
            providers._post("http://example.test", {}, {})
            assert False, "expected ProviderError"
        except providers.ProviderError as exc:
            assert exc.retry_after is None  # HTTP-date form is not parsed, not a crash


def test_call_openai_compatible_includes_extra_params(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _mock_response(
            status_code=200,
            json_body={"choices": [{"message": {"content": "[]"}}]},
        )

    monkeypatch.setattr(config, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(config, "LLM_EXTRA_PARAMS", {"reasoning_effort": "none"})
    monkeypatch.setattr(providers.requests, "post", fake_post)

    providers.call_openai_compatible("system", "user")
    assert captured["body"]["reasoning_effort"] == "none"


def test_call_openai_compatible_omits_extra_params_when_empty(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _mock_response(
            status_code=200,
            json_body={"choices": [{"message": {"content": "[]"}}]},
        )

    monkeypatch.setattr(config, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(config, "LLM_EXTRA_PARAMS", {})
    monkeypatch.setattr(providers.requests, "post", fake_post)

    providers.call_openai_compatible("system", "user")
    assert "reasoning_effort" not in captured["body"]
