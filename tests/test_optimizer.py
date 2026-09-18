"""Unit tests for gridwise.core.optimizer, cross-checked with the final
replay validator (gridwise.core.validator) so both layers are exercised
together, the way app.py uses them.
"""
import copy

import pytest

from gridwise.core import optimizer, validator


def _no_op(notes):
    return [
        {"note_index": i, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "n/a"}
        for i in range(len(notes))
    ]


def test_baseline_schedule_is_valid_and_neutral(flat_hours, minimal_battery):
    notes = ["irrelevant"]
    directives = _no_op(notes)
    result = optimizer.solve(flat_hours, minimal_battery, directives)

    violations = validator.replay_and_check(
        result["hourly_plan"], result["conditions"], directives, minimal_battery
    )
    assert violations == []
    assert result["hourly_plan"][-1]["battery_energy_after_kwh"] == pytest.approx(
        minimal_battery["initial_energy_kwh"], abs=0.01
    )


def test_no_charge_window_forces_zero_charge(flat_hours, minimal_battery):
    notes = ["Charger offline 2am-5am."]
    directives = [{
        "note_index": 0, "applies": True, "directive_type": "no_charge_window",
        "structured_adjustment": {"hours": [2, 3, 4]}, "explanation": "maintenance",
    }]
    result = optimizer.solve(flat_hours, minimal_battery, directives)

    for entry in result["hourly_plan"]:
        if entry["hour"] in (2, 3, 4):
            assert entry["battery_action"] != "charge"

    violations = validator.replay_and_check(
        result["hourly_plan"], result["conditions"], directives, minimal_battery
    )
    assert violations == []


def test_no_discharge_window_forces_zero_discharge(flat_hours, minimal_battery):
    directives = [{
        "note_index": 0, "applies": True, "directive_type": "no_discharge_window",
        "structured_adjustment": {"hours": [18, 19]}, "explanation": "relay test",
    }]
    result = optimizer.solve(flat_hours, minimal_battery, directives)

    for entry in result["hourly_plan"]:
        if entry["hour"] in (18, 19):
            assert entry["battery_action"] != "discharge"

    violations = validator.replay_and_check(
        result["hourly_plan"], result["conditions"], directives, minimal_battery
    )
    assert violations == []


def test_max_grid_window_caps_grid_import(flat_hours, minimal_battery):
    directives = [{
        "note_index": 0, "applies": True, "directive_type": "max_grid_window",
        "structured_adjustment": {"hours": [18, 19, 20], "max_grid_kwh": 60.0},
        "explanation": "feeder cap",
    }]
    result = optimizer.solve(flat_hours, minimal_battery, directives)

    for entry in result["hourly_plan"]:
        if entry["hour"] in (18, 19, 20):
            assert entry["grid_kwh"] <= 60.0 + 0.01

    violations = validator.replay_and_check(
        result["hourly_plan"], result["conditions"], directives, minimal_battery
    )
    assert violations == []


def test_minimum_battery_reserve_is_respected(flat_hours, minimal_battery):
    directives = [{
        "note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
        "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 90.0},
        "explanation": "emergency reserve",
    }]
    result = optimizer.solve(flat_hours, minimal_battery, directives)

    for entry in result["hourly_plan"]:
        if entry["hour"] in (18, 19, 20):
            assert entry["battery_energy_after_kwh"] >= 90.0 - 0.01

    violations = validator.replay_and_check(
        result["hourly_plan"], result["conditions"], directives, minimal_battery
    )
    assert violations == []


def test_solar_reduction_lowers_usable_solar():
    hours = [
        {"hour": h, "demand_kwh": 150.0, "solar_kwh": 100.0, "tariff_bdt_per_kwh": 10.0}
        for h in range(24)
    ]
    battery = {
        "capacity_kwh": 200.0, "initial_energy_kwh": 100.0, "minimum_energy_kwh": 20.0,
        "max_charge_kwh_per_hour": 50.0, "max_discharge_kwh_per_hour": 50.0,
    }
    directives = [{
        "note_index": 0, "applies": True, "directive_type": "solar_reduction",
        "structured_adjustment": {"hours": [10, 11], "factor": 0.25},
        "explanation": "cleaning",
    }]
    result = optimizer.solve(hours, battery, directives)

    for entry in result["hourly_plan"]:
        if entry["hour"] in (10, 11):
            assert entry["solar_used_kwh"] <= 25.0 + 0.01

    violations = validator.replay_and_check(
        result["hourly_plan"], result["conditions"], directives, battery
    )
    assert violations == []


def test_infeasible_scenario_raises(minimal_battery):
    # Demand impossible to cover: grid is capped to 0 all day with no solar
    # or battery headroom to make up 100 kWh/hour of demand.
    hours = [
        {"hour": h, "demand_kwh": 100.0, "solar_kwh": 0.0, "tariff_bdt_per_kwh": 10.0}
        for h in range(24)
    ]
    directives = [{
        "note_index": 0, "applies": True, "directive_type": "max_grid_window",
        "structured_adjustment": {"hours": list(range(24)), "max_grid_kwh": 0.0},
        "explanation": "total outage",
    }]
    with pytest.raises(optimizer.InfeasibleScenarioError):
        optimizer.solve(hours, minimal_battery, directives)


def test_combined_directives_all_hold(flat_hours, minimal_battery):
    directives = [
        {"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
         "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 80.0},
         "explanation": "reserve"},
        {"note_index": 1, "applies": True, "directive_type": "max_grid_window",
         "structured_adjustment": {"hours": [19, 20], "max_grid_kwh": 120.0},
         "explanation": "cap"},
    ]
    result = optimizer.solve(flat_hours, minimal_battery, directives)
    violations = validator.replay_and_check(
        result["hourly_plan"], result["conditions"], directives, minimal_battery
    )
    assert violations == []
