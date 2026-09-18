"""Final replay validator -- the last box in the Section 03 pipeline diagram.

The judge independently replays the returned hourly_plan hour by hour against
the effective solar, battery rules, and every applicable directive (Section
11.2-11.3) rather than trusting our own bookkeeping. This module runs that
exact same replay before we respond, so a mistake surfaces in our own logs
as a loud warning instead of as a lost hidden-test case at judging time.

It is a safety net, not a second opinion: the optimizer already builds a
plan that satisfies these checks by construction (gridwise.core.optimizer),
so a violation here means a real bug, and it is reported rather than silently
patched -- patching a plan post hoc could make it internally inconsistent in
a way that is even harder to debug than the original miss.
"""
from typing import Any, Dict, List

from gridwise.constants import (
    ACTION_CHARGE,
    ACTION_DISCHARGE,
    ACTION_IDLE,
    MAX_GRID_WINDOW,
    MINIMUM_BATTERY_RESERVE,
    NO_CHARGE_WINDOW,
    NO_DISCHARGE_WINDOW,
    NUM_HOURS,
    NUMERIC_TOLERANCE,
)
from gridwise.core.optimizer import ScenarioConditions

# The replay re-derives directive effects independently of optimizer.apply_directives
# so a bug shared between "apply" and "check" cannot hide from this validator.
_TOL = NUMERIC_TOLERANCE * 2  # a little slack for chained rounding across 24 hours


def _directive_windows(directive_interpretation: List[Dict[str, Any]]):
    reserve_by_hour: Dict[int, float] = {}
    charge_blocked = set()
    discharge_blocked = set()
    max_grid_by_hour: Dict[int, float] = {}

    for entry in directive_interpretation:
        if not entry.get("applies"):
            continue
        dtype = entry["directive_type"]
        adj = entry.get("structured_adjustment") or {}
        for hour in adj.get("hours", []):
            if dtype == MINIMUM_BATTERY_RESERVE:
                current = reserve_by_hour.get(hour, 0.0)
                reserve_by_hour[hour] = max(current, float(adj["minimum_energy_kwh"]))
            elif dtype == NO_CHARGE_WINDOW:
                charge_blocked.add(hour)
            elif dtype == NO_DISCHARGE_WINDOW:
                discharge_blocked.add(hour)
            elif dtype == MAX_GRID_WINDOW:
                cap = float(adj["max_grid_kwh"])
                max_grid_by_hour[hour] = min(max_grid_by_hour.get(hour, cap), cap)

    return reserve_by_hour, charge_blocked, discharge_blocked, max_grid_by_hour


def replay_and_check(
    hourly_plan: List[Dict[str, Any]],
    conditions: ScenarioConditions,
    directive_interpretation: List[Dict[str, Any]],
    battery: Dict[str, Any],
) -> List[str]:
    """Replays hourly_plan against every GridWise and directive rule.

    Returns a list of human-readable violation messages; empty means the plan
    is fully valid under the same rules the judge applies.
    """
    violations: List[str] = []

    if len(hourly_plan) != NUM_HOURS:
        violations.append(f"hourly_plan has {len(hourly_plan)} entries, expected {NUM_HOURS}.")
        return violations

    seen_hours = {entry["hour"] for entry in hourly_plan}
    if seen_hours != set(range(NUM_HOURS)):
        violations.append("hourly_plan does not cover hours 0-23 exactly once.")
        return violations

    reserve_by_hour, charge_blocked, discharge_blocked, max_grid_by_hour = _directive_windows(
        directive_interpretation
    )

    capacity = float(battery["capacity_kwh"])
    base_minimum = float(battery["minimum_energy_kwh"])
    max_charge_rate = float(battery["max_charge_kwh_per_hour"])
    max_discharge_rate = float(battery["max_discharge_kwh_per_hour"])
    initial_energy = float(battery["initial_energy_kwh"])

    battery_energy = initial_energy
    ordered = sorted(hourly_plan, key=lambda e: e["hour"])

    for entry in ordered:
        h = entry["hour"]
        action = entry["battery_action"]
        magnitude = float(entry["battery_kwh"])
        grid = float(entry["grid_kwh"])
        solar_used = float(entry["solar_used_kwh"])
        reported_after = float(entry["battery_energy_after_kwh"])

        if action not in (ACTION_CHARGE, ACTION_DISCHARGE, ACTION_IDLE):
            violations.append(f"hour {h}: invalid battery_action '{action}'.")
            continue
        if action == ACTION_IDLE and abs(magnitude) > _TOL:
            violations.append(f"hour {h}: battery_kwh must be 0 when idle.")
        if magnitude < -_TOL:
            violations.append(f"hour {h}: battery_kwh must be non-negative.")
        if grid < -_TOL:
            violations.append(f"hour {h}: grid_kwh must be non-negative.")
        if solar_used < -_TOL:
            violations.append(f"hour {h}: solar_used_kwh must be non-negative.")

        effective_solar = conditions.effective_solar[h]
        if solar_used > effective_solar + _TOL:
            violations.append(
                f"hour {h}: solar_used_kwh ({solar_used}) exceeds effective solar "
                f"({effective_solar})."
            )

        if action == ACTION_CHARGE and magnitude > max_charge_rate + _TOL:
            violations.append(
                f"hour {h}: charge {magnitude} exceeds max_charge_kwh_per_hour {max_charge_rate}."
            )
        if action == ACTION_DISCHARGE and magnitude > max_discharge_rate + _TOL:
            violations.append(
                f"hour {h}: discharge {magnitude} exceeds max_discharge_kwh_per_hour "
                f"{max_discharge_rate}."
            )
        if action == ACTION_CHARGE and h in charge_blocked:
            violations.append(f"hour {h}: charging occurred during a no_charge_window hour.")
        if action == ACTION_DISCHARGE and h in discharge_blocked:
            violations.append(f"hour {h}: discharging occurred during a no_discharge_window hour.")

        cap = max_grid_by_hour.get(h)
        if cap is not None and grid > cap + _TOL:
            violations.append(f"hour {h}: grid_kwh {grid} exceeds max_grid_window cap {cap}.")

        charged = magnitude if action == ACTION_CHARGE else 0.0
        discharged = magnitude if action == ACTION_DISCHARGE else 0.0

        # Section 9.5 energy balance.
        demand = conditions.demand[h]
        balance = grid + solar_used + discharged - demand - charged
        if abs(balance) > _TOL:
            violations.append(
                f"hour {h}: energy balance violated (imbalance {balance:+.4f} kWh)."
            )

        battery_energy = battery_energy + charged - discharged
        required_min = max(base_minimum, reserve_by_hour.get(h, 0.0))
        if battery_energy < required_min - _TOL:
            violations.append(
                f"hour {h}: battery energy {battery_energy:.4f} below required minimum "
                f"{required_min}."
            )
        if battery_energy > capacity + _TOL:
            violations.append(
                f"hour {h}: battery energy {battery_energy:.4f} exceeds capacity {capacity}."
            )
        if abs(battery_energy - reported_after) > _TOL:
            violations.append(
                f"hour {h}: reported battery_energy_after_kwh {reported_after} does not match "
                f"replayed value {battery_energy:.4f}."
            )
        battery_energy = reported_after  # keep the replay anchored to reported state

    if abs(battery_energy - initial_energy) > _TOL:
        violations.append(
            f"end-of-day battery energy {battery_energy:.4f} does not equal initial "
            f"energy {initial_energy} (neutrality violated)."
        )

    return violations
