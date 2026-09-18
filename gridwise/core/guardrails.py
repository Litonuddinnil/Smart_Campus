"""Deterministic guardrail layer (Problem Statement, Section 08).

LLM output is untrusted. This module is the single place that decides whether
a raw interpretation entry is safe to hand to the optimizer. Any entry that
fails validation is deterministically replaced with a safe no_op -- per the
SAFE FAILURE rule, the service must never crash and never invent an
unsupported directive.

Invariant this module guarantees to every caller: the returned list has
exactly len(operator_notes) entries, note_index 0..N-1 in ascending order,
each one schema-correct for its directive_type.
"""
from typing import Any, Dict, List, Optional

from gridwise.constants import (
    ALLOWED_DIRECTIVE_TYPES,
    MAX_GRID_WINDOW,
    MINIMUM_BATTERY_RESERVE,
    NO_CHARGE_WINDOW,
    NO_DISCHARGE_WINDOW,
    NO_OP,
    SOLAR_REDUCTION,
)


def _safe_no_op(note_index: int, reason: str) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": NO_OP,
        "structured_adjustment": None,
        "explanation": f"Guardrail fallback: {reason}",
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def valid_hours_list(hours: Any) -> bool:
    """Section 08 `Hours` guardrail: unique ints 0-23, ascending, non-empty."""
    if not isinstance(hours, list) or not hours:
        return False
    if not all(_is_int(h) and 0 <= h <= 23 for h in hours):
        return False
    if len(set(hours)) != len(hours):
        return False
    return hours == sorted(hours)


def _validate_adjustment(
    directive_type: str,
    adjustment: Any,
    battery_capacity: Optional[float] = None,
) -> bool:
    """Checks structured_adjustment against the Section 04.1 required shape.

    Extra keys are tolerated; missing or out-of-range required keys are not.
    """
    if not isinstance(adjustment, dict):
        return False

    if directive_type == SOLAR_REDUCTION:
        if "hours" not in adjustment or "factor" not in adjustment:
            return False
        if not valid_hours_list(adjustment["hours"]):
            return False
        if not _is_number(adjustment["factor"]):
            return False
        return 0.0 <= float(adjustment["factor"]) <= 1.0

    if directive_type == MINIMUM_BATTERY_RESERVE:
        if "hours" not in adjustment or "minimum_energy_kwh" not in adjustment:
            return False
        if not valid_hours_list(adjustment["hours"]):
            return False
        reserve = adjustment["minimum_energy_kwh"]
        if not _is_number(reserve) or reserve < 0:
            return False
        # Section 08: a reserve above capacity is unsatisfiable, so reject it
        # rather than handing the optimizer a guaranteed-infeasible problem.
        if battery_capacity is not None and reserve > battery_capacity:
            return False
        return True

    if directive_type in (NO_CHARGE_WINDOW, NO_DISCHARGE_WINDOW):
        return "hours" in adjustment and valid_hours_list(adjustment["hours"])

    if directive_type == MAX_GRID_WINDOW:
        if "hours" not in adjustment or "max_grid_kwh" not in adjustment:
            return False
        if not valid_hours_list(adjustment["hours"]):
            return False
        cap = adjustment["max_grid_kwh"]
        return _is_number(cap) and cap >= 0

    return False


def validate_and_clean(
    raw_entries: Any,
    operator_notes: List[str],
    battery_capacity: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Turns raw model output into a guaranteed schema-correct directive list.

    Missing, duplicate, out-of-range, or malformed entries each degrade to a
    safe no_op for that note rather than failing the whole request.
    """
    note_count = len(operator_notes)
    by_index: Dict[int, Dict[str, Any]] = {}

    if isinstance(raw_entries, list):
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            index = entry.get("note_index")
            if not _is_int(index) or not (0 <= index < note_count):
                continue
            if index in by_index:
                # Duplicate mapping for one note: keep the first, drop the rest.
                continue
            by_index[index] = entry

    cleaned: List[Dict[str, Any]] = []
    for index in range(note_count):
        entry = by_index.get(index)
        if entry is None:
            cleaned.append(_safe_no_op(index, "no interpretation returned for this note"))
            continue

        directive_type = entry.get("directive_type")
        explanation = entry.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = "No explanation provided."

        if directive_type not in ALLOWED_DIRECTIVE_TYPES:
            cleaned.append(_safe_no_op(index, "unsupported or missing directive_type"))
            continue

        if directive_type == NO_OP:
            # Section 05.1: no_op is always applies=false with a null adjustment,
            # whatever the model put in those fields.
            cleaned.append({
                "note_index": index,
                "applies": False,
                "directive_type": NO_OP,
                "structured_adjustment": None,
                "explanation": explanation,
            })
            continue

        if entry.get("applies") is not True:
            cleaned.append(_safe_no_op(index, "non-no_op directive missing applies=true"))
            continue

        adjustment = entry.get("structured_adjustment")
        if not _validate_adjustment(directive_type, adjustment, battery_capacity):
            cleaned.append(_safe_no_op(index, f"invalid structured_adjustment for {directive_type}"))
            continue

        cleaned.append({
            "note_index": index,
            "applies": True,
            "directive_type": directive_type,
            "structured_adjustment": adjustment,
            "explanation": explanation,
        })

    return cleaned


def safe_fallback(operator_notes: List[str], reason: str) -> List[Dict[str, Any]]:
    """Every note becomes a no_op -- used when the model call fails outright.

    The response stays valid and schema-correct; only interpretation credit is
    lost, which beats returning a 5xx and losing the case entirely.
    """
    return [_safe_no_op(i, reason) for i in range(len(operator_notes))]
