"""Unit tests for gridwise.api.schema (Problem Statement, Section 07)."""
import copy

import pytest

from gridwise.api import schema


def _valid_payload(flat_hours, minimal_battery):
    return {
        "scenario_id": "TEST-1",
        "operator_notes": ["Do not charge from 2pm to 4pm."],
        "hours": copy.deepcopy(flat_hours),
        "battery": copy.deepcopy(minimal_battery),
    }


def test_valid_payload_passes(flat_hours, minimal_battery):
    payload = _valid_payload(flat_hours, minimal_battery)
    schema.validate_request(payload)  # should not raise


def test_missing_scenario_id_rejected(flat_hours, minimal_battery):
    payload = _valid_payload(flat_hours, minimal_battery)
    del payload["scenario_id"]
    with pytest.raises(schema.ValidationError):
        schema.validate_request(payload)


def test_too_many_operator_notes_rejected(flat_hours, minimal_battery):
    payload = _valid_payload(flat_hours, minimal_battery)
    payload["operator_notes"] = ["a", "b", "c", "d"]
    with pytest.raises(schema.ValidationError):
        schema.validate_request(payload)


def test_empty_operator_notes_rejected(flat_hours, minimal_battery):
    payload = _valid_payload(flat_hours, minimal_battery)
    payload["operator_notes"] = []
    with pytest.raises(schema.ValidationError):
        schema.validate_request(payload)


def test_wrong_hour_count_rejected(flat_hours, minimal_battery):
    payload = _valid_payload(flat_hours, minimal_battery)
    payload["hours"] = payload["hours"][:23]
    with pytest.raises(schema.ValidationError):
        schema.validate_request(payload)


def test_duplicate_hour_rejected(flat_hours, minimal_battery):
    payload = _valid_payload(flat_hours, minimal_battery)
    payload["hours"][1]["hour"] = 0
    with pytest.raises(schema.ValidationError):
        schema.validate_request(payload)


def test_negative_demand_rejected(flat_hours, minimal_battery):
    payload = _valid_payload(flat_hours, minimal_battery)
    payload["hours"][0]["demand_kwh"] = -5
    with pytest.raises(schema.ValidationError):
        schema.validate_request(payload)


def test_battery_reserve_above_capacity_rejected(flat_hours, minimal_battery):
    payload = _valid_payload(flat_hours, minimal_battery)
    payload["battery"]["minimum_energy_kwh"] = 9999
    with pytest.raises(schema.ValidationError):
        schema.validate_request(payload)


def test_battery_initial_below_minimum_rejected(flat_hours, minimal_battery):
    payload = _valid_payload(flat_hours, minimal_battery)
    payload["battery"]["initial_energy_kwh"] = 0
    payload["battery"]["minimum_energy_kwh"] = 30
    with pytest.raises(schema.ValidationError):
        schema.validate_request(payload)


def test_normalize_hours_sorts_ascending():
    hours = [{"hour": 2}, {"hour": 0}, {"hour": 1}]
    normalized = schema.normalize_hours(hours)
    assert [h["hour"] for h in normalized] == [0, 1, 2]


def test_non_dict_body_rejected():
    with pytest.raises(schema.ValidationError):
        schema.validate_request(["not", "a", "dict"])
