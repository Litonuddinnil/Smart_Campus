# Architecture

Written for: organizers/judges reviewing the submission, and future maintainers of this repo.

## Pipeline

```
Energy Data + Operator Notes
        |
        v
  LLM Interpreter        gridwise/llm/            (untrusted output)
        |
        v
  Guardrail Validator    gridwise/core/guardrails.py
        |
        v
  Math Optimizer         gridwise/core/optimizer.py
        |
        v
  Final Validator        gridwise/core/validator.py
        |
        v
  API Response           gridwise/api/app.py
```

This mirrors the Problem Statement's Section 03 diagram exactly. Each stage
has one job and trusts nothing from the stage before it except what that
stage's own contract guarantees.

## Why the LLM's output is never trusted directly

`gridwise/llm/interpreter.py` calls a language model and does only enough
parsing to get *something* structured out of free text (stripping markdown
fences, unwrapping `{"directives": [...]}`, etc). It never checks whether the
result is *correct*.

`gridwise/core/guardrails.py` is the only place that decides whether an
interpretation entry is safe to act on. For every operator note it either:

- keeps the model's directive as-is (type is supported, hours are unique
  ascending integers 0-23, the structured_adjustment shape matches, numeric
  ranges are sane), or
- replaces it with a safe `no_op` and records why.

The function's contract is unconditional: given *any* input (a list, `None`,
a string, garbage, duplicate note_index values, an out-of-range note_index),
it always returns exactly `len(operator_notes)` entries, in order, each one
schema-correct. That is what makes "the model returned nonsense" a lost
interpretation point instead of a 500 error or a fabricated energy rule.

## Why the optimizer is a real LP, not a heuristic

`gridwise/core/optimizer.py` builds a 96-variable linear program (`grid`,
`solar_used`, `charge`, `discharge` per hour) and solves it with
`scipy.optimize.linprog` (HiGHS backend). Every directive from Section 04
becomes either a variable bound (charge/discharge blocked, solar ceiling,
grid cap) or a constraint (battery reserve via prefix sums over
charge/discharge). Solving takes low-single-digit milliseconds on one CPU
core, which is what keeps this comfortably inside the judge's 30s per-request
budget on a free CPU-only host with no GPU.

Because it is exact, not a greedy or rule-based heuristic, cross-checking it
against all 10 organizer public sample cases (`tests/test_public_cases.py`)
shows its cost matching or beating the published reference schedule on every
case -- an LP cannot do worse than a hand-built optimal schedule once it is
given the same constraints.

## Why there is a final validator

The Problem Statement's diagram includes a "Final Validator" box after the
optimizer, and Section 11 describes the judge independently replaying the
returned `hourly_plan` hour by hour. `gridwise/core/validator.py` runs that
exact same replay -- energy balance, effective solar, battery bounds and
rate limits, every applicable directive, end-of-day neutrality -- before the
response is ever sent.

The optimizer already builds a plan that satisfies these checks by
construction, so a violation here means a real bug in our own code, not bad
input. It is logged loudly (`logger.error`) rather than silently patched:
patching a plan after the fact risks making it internally inconsistent in a
way that is *harder* to debug than the original miss, and the whole point of
this stage is to catch that class of bug in our own logs instead of in the
judge's hidden test results.

## Why providers are pluggable

`gridwise/config.py` defines a small preset table (`PROVIDER_PRESETS`)
mapping a provider name to a base URL, a default model, and a wire format
(`openai`-compatible or `anthropic`). Every preset except Anthropic speaks
the same OpenAI chat-completions JSON shape, so `gridwise/llm/providers.py`
needs only two transport functions total to cover OpenAI, Groq, Gemini,
Hugging Face Inference Providers, Cerebras, OpenRouter, Ollama, and any other
OpenAI-compatible endpoint (`LLM_PROVIDER=custom` + `LLM_BASE_URL`).

Switching providers is an environment-variable change, never a code change --
which matters for a submission whose LLM dependency has to survive judging on
someone else's free-tier quota.

## Package layout

```
gridwise/
├── config.py           environment-driven runtime configuration
├── constants.py         spec-fixed values (directive types, tolerance, ...)
├── llm/                 the one non-deterministic pipeline stage
│   ├── prompt.py         system + user prompt construction
│   ├── providers.py      HTTP transport per wire format
│   └── interpreter.py    orchestration, parsing, caching, retries
├── core/                 deterministic pipeline stages
│   ├── guardrails.py      untrusted -> trusted directive_interpretation
│   ├── optimizer.py       the LP solver
│   └── validator.py       final independent replay check
└── api/                  HTTP layer
    ├── schema.py          request validation
    └── app.py             Flask routes + error handling
```
