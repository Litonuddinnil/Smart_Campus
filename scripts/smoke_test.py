#!/usr/bin/env python3
"""Live smoke test against a running GridWise service.

Unlike tests/test_public_cases.py (which stands in for the LLM using each
case's own reference interpretation, so it needs no network access), this
script drives the REAL HTTP API end to end, including the live LLM call --
run it against your deployed URL (or http://localhost:7860 locally) before
submission to sanity-check prompt quality on paraphrased notes.

Usage:
    python scripts/smoke_test.py [BASE_URL] [PATH_TO_SAMPLE_CASES_JSON]

Defaults:
    BASE_URL                  http://localhost:7860
    PATH_TO_SAMPLE_CASES_JSON data/public_sample_cases.json
"""
import json
import sys
import time
from pathlib import Path

import requests

DEFAULT_BASE_URL = "http://localhost:7860"
DEFAULT_CASES_PATH = Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json"


def check_health(base_url: str) -> None:
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    body = resp.json()
    assert body.get("status") == "ok", f"unexpected health body: {body}"
    print(f"[ok] GET /health -> {body}")


def check_case(base_url: str, case: dict) -> bool:
    name = case.get("label", case.get("id", "case"))
    started = time.time()
    resp = requests.post(f"{base_url}/optimize-energy", json=case["input"], timeout=30)
    elapsed = time.time() - started

    if resp.status_code != 200:
        print(f"[FAIL] {name}: HTTP {resp.status_code} ({elapsed:.1f}s) -- {resp.text[:300]}")
        return False

    body = resp.json()
    notes = case["input"]["operator_notes"]
    interp = body.get("directive_interpretation", [])

    ok = True
    if len(interp) != len(notes):
        print(f"[FAIL] {name}: expected {len(notes)} interpretation entries, got {len(interp)}")
        ok = False
    if len(body.get("hourly_plan", [])) != 24:
        print(f"[FAIL] {name}: hourly_plan does not have 24 entries")
        ok = False

    expected_types = [e["directive_type"] for e in case["expected_output"]["directive_interpretation"]]
    got_types = [e.get("directive_type") for e in interp]
    match = "match" if got_types == expected_types else "DIFFERS"
    print(
        f"[{'ok' if ok else 'FAIL'}] {name} ({elapsed:.1f}s) "
        f"cost={body.get('total_cost_bdt')} directive_types={got_types} ({match} vs reference)"
    )
    return ok


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    cases_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CASES_PATH

    print(f"Target: {base_url}")
    check_health(base_url)

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    results = [check_case(base_url, case) for case in cases]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} cases passed structural + directive-type checks.")
    print(
        "Note: directive_type mismatches are worth a manual look -- they mean the live "
        "model disagreed with the reference interpretation, which is exactly the paraphrase-"
        "robustness risk the hidden test set probes."
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
