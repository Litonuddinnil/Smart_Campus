"""Shared pytest fixtures."""
import json
import os

import pytest

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "public_sample_cases.json")


@pytest.fixture(scope="session")
def public_cases():
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


@pytest.fixture()
def minimal_battery():
    return {
        "capacity_kwh": 200.0,
        "initial_energy_kwh": 100.0,
        "minimum_energy_kwh": 30.0,
        "max_charge_kwh_per_hour": 50.0,
        "max_discharge_kwh_per_hour": 50.0,
    }


@pytest.fixture()
def flat_hours():
    """24 hours of constant demand/tariff and no solar -- the simplest
    feasible scenario, useful when a test only cares about directive wiring.
    """
    return [
        {"hour": h, "demand_kwh": 100.0, "solar_kwh": 0.0, "tariff_bdt_per_kwh": 10.0}
        for h in range(24)
    ]
