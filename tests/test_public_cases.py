"""End-to-end test against the organizer's public sample cases.

Each case's own `expected_output.directive_interpretation` stands in for
"what the LLM should have produced," so this suite proves the guardrail,
optimizer, and final-validator layers are correct without requiring a live
network call to a model provider -- useful offline, in CI, and for exactly
this sandbox, which has no outbound network access.

llm_interpreter.interpret_notes() is the only piece not exercised here; wire
a real LLM_API_KEY and run scripts/smoke_test.py against a live provider
before submission to check prompt quality on paraphrased notes.
"""
import pytest

from gridwise.api import schema
from gridwise.core import guardrails, optimizer, validator

# The judge's own tolerance is 0.01; this suite allows a little more slack to
# absorb the harness's independent re-derivation of totals from a fresh LP
# solve, which can land on an equally optimal but differently-rounded plan.
COST_TOLERANCE_RATIO = 1.02


def _case_ids(cases):
    return [c.get("label", c.get("id", "case")) for c in cases]


def test_public_case_pack_has_ten_cases(public_cases):
    assert len(public_cases) == 10


@pytest.mark.parametrize("case_index", range(10))
def test_public_case_end_to_end(public_cases, case_index):
    case = public_cases[case_index]
    req = case["input"]
    expected = case["expected_output"]

    # 1. Request schema must validate cleanly.
    schema.validate_request(req)
    hours = schema.normalize_hours(req["hours"])
    battery = req["battery"]
    operator_notes = req["operator_notes"]

    # 2. Feed the case's reference interpretation through guardrails exactly
    #    as app.py feeds real (untrusted) LLM output through them.
    stand_in_llm_output = expected["directive_interpretation"]
    directive_interpretation = guardrails.validate_and_clean(
        stand_in_llm_output, operator_notes, battery_capacity=battery.get("capacity_kwh")
    )

    assert len(directive_interpretation) == len(operator_notes)
    for i, entry in enumerate(directive_interpretation):
        assert entry["note_index"] == i

    # Guardrails must not have downgraded a reference directive to no_op --
    # if they did, either the reference shape changed or a guardrail is
    # stricter than the spec allows.
    for expected_entry, cleaned_entry in zip(stand_in_llm_output, directive_interpretation):
        assert cleaned_entry["directive_type"] == expected_entry["directive_type"], (
            f"guardrails downgraded note {expected_entry['note_index']} "
            f"from {expected_entry['directive_type']} to {cleaned_entry['directive_type']}"
        )
        assert cleaned_entry["applies"] == expected_entry["applies"]

    # 3. Optimizer must produce a schedule that the final validator accepts.
    result = optimizer.solve(hours, battery, directive_interpretation)
    violations = validator.replay_and_check(
        result["hourly_plan"], result["conditions"], directive_interpretation, battery
    )
    assert violations == [], f"final validator found violations: {violations}"

    # 4. hourly_plan covers all 24 hours and matches recalculated totals.
    assert len(result["hourly_plan"]) == 24
    recalculated_total = sum(
        e["grid_kwh"] * hours[e["hour"]]["tariff_bdt_per_kwh"] for e in result["hourly_plan"]
    )
    assert result["total_cost_bdt"] == pytest.approx(recalculated_total, abs=0.05)

    # 5. Our LP is a true optimizer, so its cost should be at or below the
    #    reference cost (within tolerance) -- never meaningfully worse.
    expected_cost = expected["total_cost_bdt"]
    assert result["total_cost_bdt"] <= expected_cost * COST_TOLERANCE_RATIO, (
        f"optimizer cost {result['total_cost_bdt']} exceeds reference "
        f"{expected_cost} by more than {COST_TOLERANCE_RATIO}x"
    )
