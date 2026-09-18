---
title: GridWise API
emoji: 🔋
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# GridWise API — BUP CSE Fest 2026 Hackathon (Online Preliminary)

An LLM-assisted campus energy scheduling service. It interprets natural-language
operator notes, deterministically validates the extracted directives, solves a
true linear program for the lowest-cost valid 24-hour grid/solar/battery
schedule, and independently replays that schedule before responding.

Runs on a single CPU core (no GPU required) and is designed to deploy for
free on a Hugging Face Space — see [Deploying to Hugging Face Spaces](#deploying-to-hugging-face-spaces).

## Architecture

```
Energy Data + Operator Notes -> LLM Interpreter -> Guardrail Validator -> Math Optimizer -> Final Validator -> API Response
```

- **LLM Interpreter** (`gridwise/llm/`) — sends every operator note to a
  configured language model and asks it to map each one to one of the six
  supported directive types, or `no_op`. Output is treated as **untrusted**.
  Provider is fully pluggable via environment variables (see below) — no
  code change needed to switch.
- **Guardrail Validator** (`gridwise/core/guardrails.py`) — deterministic
  Python that checks directive type, note-index mapping, hour ranges
  (unique, ascending, 0–23), numeric ranges (e.g. `factor` in `[0,1]`), and
  `applies` semantics. Anything invalid is safely replaced with a `no_op`
  entry — the service never crashes and never invents an unsupported
  directive.
- **Math Optimizer** (`gridwise/core/optimizer.py`) — builds a linear program
  (96 variables: grid/solar_used/charge/discharge per hour) with
  `scipy.optimize.linprog` (HiGHS solver) and finds the true cost-minimizing
  schedule subject to energy balance, battery bounds, charge/discharge rate
  limits, and every applicable directive. This is exact optimization, not a
  heuristic, and it solves in single-digit milliseconds on one CPU core.
- **Final Validator** (`gridwise/core/validator.py`) — independently replays
  the completed schedule hour by hour (energy balance, effective solar,
  battery bounds/rate limits, every directive, end-of-day neutrality) before
  it is returned, the same way the judge harness will. A violation here means
  our own bug, not bad input, so it is logged loudly instead of silently
  patched.
- **API layer** (`gridwise/api/`) — Flask app exposing `GET /health` and
  `POST /optimize-energy`, with hand-written request validation (Flask is the
  only framework dependency) and controlled error responses (400/422/500).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design
rationale behind each stage.

## Model / provider used

Configurable entirely via environment variables — no code change needed to
switch provider or model:

| Env var | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `groq` \| `gemini` \| `huggingface` \| `cerebras` \| `openrouter` \| `openai` \| `anthropic` \| `ollama` \| `custom` | `groq` |
| `LLM_API_KEY` | API key for the selected provider (or set the provider's own conventional var, e.g. `GROQ_API_KEY`, `HF_TOKEN`, `GEMINI_API_KEY`) | *(required except `ollama`/`custom` with no auth)* |
| `LLM_MODEL` | Model id | provider-specific default, see `gridwise/config.py` |
| `LLM_BASE_URL` | Base URL — only needed for `LLM_PROVIDER=custom` | provider preset |
| `LLM_TIMEOUT_SECONDS` | Per-call timeout | `12` |
| `LLM_MAX_RETRIES` | Retries before failing safe | `1` |
| `LLM_CACHE_ENABLED` | Cache identical interpretations in memory | `true` |
| `HOST` / `PORT` | Bind address | `0.0.0.0` / `7860` |

**Default provider is Groq** because it has a genuinely free API tier fast
enough to stay well inside the judge's 30s-per-request / 5s-p95 budget. Any
OpenAI-compatible endpoint works by setting `LLM_PROVIDER=custom` and
`LLM_BASE_URL`; Anthropic is supported natively.

If the model call fails entirely (bad key, provider outage, timeout after
retries), every note is safely marked `no_op` rather than crashing the
service — the API still returns a valid, schema-correct response (interpretation
credit is lost for that request, but the request does not fail).

## Local quickstart (clean environment)

```bash
git clone <your-repo-url>
cd "Smart Campus"
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER and LLM_API_KEY (a free Groq key works out of the box)

python3 app.py
```

Service starts on `http://0.0.0.0:7860`.

### Health check

```bash
curl http://localhost:7860/health
# {"status":"ok"}
```

### Sample optimize-energy request

```bash
curl -X POST http://localhost:7860/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Any single `case.input` object from [`data/public_sample_cases.json`](data/public_sample_cases.json)
is a valid `sample_request.json`. Example (trimmed):

```json
{
  "scenario_id": "SAMPLE-02",
  "operator_notes": ["The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance."],
  "hours": [ { "hour": 0, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 6 }, "... 23 more ..." ],
  "battery": { "capacity_kwh": 200, "initial_energy_kwh": 70, "minimum_energy_kwh": 30, "max_charge_kwh_per_hour": 55, "max_discharge_kwh_per_hour": 55 }
}
```

Expected shape of the response (values omitted for brevity):

```json
{
  "scenario_id": "SAMPLE-02",
  "directive_interpretation": [ { "note_index": 0, "applies": true, "directive_type": "no_charge_window", "structured_adjustment": {"hours": [2,3,4]}, "explanation": "..." } ],
  "hourly_plan": [ { "hour": 0, "grid_kwh": 120.0, "solar_used_kwh": 0.0, "battery_action": "charge", "battery_kwh": 20.0, "battery_energy_after_kwh": 90.0 }, "... 23 more ..." ],
  "total_grid_kwh": 2915.0,
  "total_cost_bdt": 42885.0,
  "peak_grid_kwh": 180.0,
  "plan_summary": "..."
}
```

### Run the automated test suite

```bash
pip install -r requirements-dev.txt
pytest -v
```

53 tests cover request-schema validation, guardrail behavior on malformed/
adversarial model output, the LP optimizer against 8 directive scenarios
(including an intentionally infeasible one), LLM response parsing/caching
with the network mocked out, and — the most important one —
**every one of the 10 organizer public sample cases end to end**: schema →
guardrails → optimizer → final validator, checking that the returned
schedule is fully valid and that its cost matches or beats the published
reference. This does not require a live LLM call (see
[`tests/test_public_cases.py`](tests/test_public_cases.py)'s docstring for
why, and how it still exercises the guardrail/optimizer/validator layers
exactly as the live app does).

### Live smoke test (needs a real API key)

```bash
python app.py &                 # or point at your deployed URL
python scripts/smoke_test.py http://localhost:7860 data/public_sample_cases.json
```

This drives the real HTTP API, including the live model call, and flags any
case where the live interpretation disagrees with the reference directive
type — the exact risk the hidden paraphrased test set probes. Run it once
against your deployed URL before submission.

## Deploying to Hugging Face Spaces

This repo is ready to deploy as a **Docker-SDK Space** with no GPU:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space),
   SDK = **Docker**, hardware = the free **CPU basic** tier.
2. Push this repository's contents to the Space's git remote (Spaces are git
   repos):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space>
   git push space main
   ```
3. In the Space's **Settings → Variables and secrets**, add:
   - `LLM_PROVIDER` = `groq` (or your chosen provider)
   - `LLM_API_KEY` = your key, added as a **secret**, never as plain a variable
4. The Space builds the `Dockerfile` and starts the container automatically.
   It already exposes port `7860` (the Spaces default) and binds to
   `0.0.0.0`, matching the `app_port: 7860` declared in this README's YAML
   header above.
5. Once it's live, your judge-facing base URL is
   `https://<your-username>-<your-space>.hf.space`. Verify with:
   ```bash
   curl https://<your-username>-<your-space>.hf.space/health
   ```

The optimizer (`scipy.optimize.linprog`, HiGHS) and the guardrail/validator
layers are pure CPU code with no heavy ML dependency, so the free CPU-only
tier is sufficient — the only network call the service makes per request is
the one LLM interpretation call.

## Docker fallback (organizer-run, without Hugging Face)

```bash
docker build -t gridwise:latest .
docker run -p 7860:7860 \
  -e LLM_PROVIDER=groq \
  -e LLM_API_KEY=REPLACE-ME \
  gridwise:latest
```

Exposes port `7860`, binds to `0.0.0.0`. No secrets are baked into the image —
all credentials are supplied at `docker run` time.

## Dependencies

- **Flask** — HTTP server
- **scipy** — `linprog` (HiGHS) for the LP optimizer
- **numpy** — constraint-matrix construction
- **requests** — outbound LLM API calls
- **gunicorn** — production WSGI server (used in Docker/Spaces; `python3 app.py` uses Flask's dev server for local runs)
- **pytest** (dev only, `requirements-dev.txt`) — test suite

## Known limitations

- Only OpenAI-compatible providers (Groq, Gemini, Hugging Face Inference
  Providers, Cerebras, OpenRouter, OpenAI, Ollama, or any other
  OpenAI-compatible endpoint via `LLM_PROVIDER=custom`) and Anthropic are
  wired up. Adding a genuinely different wire format means adding one
  function in `gridwise/llm/providers.py`.
- The optimizer assumes directive combinations compose without hard
  contradictions, per the Problem Statement's guarantee that organizer-valid
  scenarios are feasible; a genuinely infeasible combination returns HTTP 422
  with an explanation rather than a fabricated schedule.
- `tests/test_public_cases.py` validates the guardrail/optimizer/validator
  layers using each sample case's own expected interpretation as a stand-in
  for the LLM call (so the suite needs no live API key); it does not by
  itself prove prompt quality against paraphrased hidden notes — run
  `scripts/smoke_test.py` against a live deployment before submission to
  check that.
- The in-memory interpretation cache (`LLM_CACHE_ENABLED`) is per-process; it
  resets on restart and is not shared across multiple gunicorn workers. This
  is intentional — the Dockerfile runs a single worker, so no cross-worker
  cache inconsistency is possible.

## Directive types implemented

| Type | Effect on optimizer |
|---|---|
| `solar_reduction` | `effective_solar[h] *= factor` |
| `minimum_battery_reserve` | raises battery lower bound for listed hours |
| `no_charge_window` | forces `charge[h] = 0` |
| `no_discharge_window` | forces `discharge[h] = 0` |
| `max_grid_window` | caps `grid_kwh[h]` |
| `no_op` | no effect |

## Repository layout

```
.
├── app.py                    # root entrypoint (gunicorn/Spaces target: app:app)
├── gridwise/                 # the service, as an importable package
│   ├── config.py              # environment-driven runtime configuration
│   ├── constants.py           # spec-fixed values (directive types, tolerance)
│   ├── llm/                   # LLM interpretation (untrusted)
│   ├── core/                  # guardrails, LP optimizer, final validator
│   └── api/                   # Flask routes + request schema
├── tests/                     # pytest suite (unit + public-case end-to-end)
├── scripts/smoke_test.py      # live HTTP smoke test against a deployed URL
├── data/public_sample_cases.json   # organizer's public sample cases (copied in)
├── docs/ARCHITECTURE.md       # design rationale for each pipeline stage
├── Dockerfile                 # CPU-only, port 7860, no baked-in secrets
├── requirements.txt           # runtime dependencies
├── requirements-dev.txt       # + pytest, for local/CI use only
└── .env.example                # documents every environment variable
```
