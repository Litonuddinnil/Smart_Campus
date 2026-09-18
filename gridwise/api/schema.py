"""Request validation for POST /optimize-energy (Problem Statement, Section 07).

Hand-written rather than pydantic/marshmallow so the validation layer itself
has zero extra dependency -- one less thing to break on a constrained free
host, and one less version to pin.
"""
from typing import Any, Dict, List

from gridwise.constants import NUM_HOURS


class ValidationError(Exception):
    """The request body does not match the required schema."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_request(payload: Dict[str, Any]) -> None:
    """Validates the full request body. Raises ValidationError on the first
    violation found, with a human-readable message safe to return to the caller.
    """
    _require(isinstance(payload, dict), "Request body must be a JSON object.")

    _require("scenario_id" in payload, "Missing required field: scenario_id.")
    scenario_id = payload["scenario_id"]
    _require(
        isinstance(scenario_id, str) and scenario_id.strip() != "",
        "scenario_id must be a non-empty string.",
    )

    _require("operator_notes" in payload, "Missing required field: operator_notes.")
    notes = payload["operator_notes"]
    _require(isinstance(notes, list), "operator_notes must be an array.")
    _require(1 <= len(notes) <= 3, "operator_notes must contain between 1 and 3 entries.")
    for i, note in enumerate(notes):
        _require(
            isinstance(note, str) and note.strip() != "",
            f"operator_notes[{i}] must be a non-empty string.",
        )

    _require("hours" in payload, "Missing required field: hours.")
    hours = payload["hours"]
    _require(
        isinstance(hours, list) and len(hours) == NUM_HOURS,
        f"hours must be an array of exactly {NUM_HOURS} entries.",
    )

    seen_hours = set()
    for i, entry in enumerate(hours):
        _require(isinstance(entry, dict), f"hours[{i}] must be an object.")
        for field in ("hour", "demand_kwh", "solar_kwh", "tariff_bdt_per_kwh"):
            _require(field in entry, f"hours[{i}] missing required field: {field}.")

        _require(_is_int(entry["hour"]), f"hours[{i}].hour must be an integer.")
        _require(0 <= entry["hour"] <= 23, f"hours[{i}].hour must be between 0 and 23.")
        _require(entry["hour"] not in seen_hours, f"Duplicate hour value: {entry['hour']}.")
        seen_hours.add(entry["hour"])

        _require(
            _is_number(entry["demand_kwh"]) and entry["demand_kwh"] >= 0,
            f"hours[{i}].demand_kwh must be a non-negative number.",
        )
        _require(
            _is_number(entry["solar_kwh"]) and entry["solar_kwh"] >= 0,
            f"hours[{i}].solar_kwh must be a non-negative number.",
        )
        _require(
            _is_number(entry["tariff_bdt_per_kwh"]) and entry["tariff_bdt_per_kwh"] >= 0,
            f"hours[{i}].tariff_bdt_per_kwh must be a non-negative number.",
        )

    _require(
        seen_hours == set(range(NUM_HOURS)),
        f"hours must cover every integer from 0 to {NUM_HOURS - 1} exactly once.",
    )

    _require("battery" in payload, "Missing required field: battery.")
    battery = payload["battery"]
    _require(isinstance(battery, dict), "battery must be an object.")
    for field in (
        "capacity_kwh",
        "initial_energy_kwh",
        "minimum_energy_kwh",
        "max_charge_kwh_per_hour",
        "max_discharge_kwh_per_hour",
    ):
        _require(field in battery, f"battery missing required field: {field}.")
        _require(
            _is_number(battery[field]) and battery[field] >= 0,
            f"battery.{field} must be a non-negative number.",
        )

    _require(
        battery["minimum_energy_kwh"] <= battery["capacity_kwh"],
        "battery.minimum_energy_kwh cannot exceed battery.capacity_kwh.",
    )
    _require(
        battery["initial_energy_kwh"] <= battery["capacity_kwh"],
        "battery.initial_energy_kwh cannot exceed battery.capacity_kwh.",
    )
    _require(
        battery["initial_energy_kwh"] >= battery["minimum_energy_kwh"],
        "battery.initial_energy_kwh cannot be below battery.minimum_energy_kwh.",
    )


def normalize_hours(hours: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Returns hour entries sorted ascending by `hour` (0..23)."""
    return sorted(hours, key=lambda entry: entry["hour"])
