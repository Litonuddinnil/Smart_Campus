"""Linear-programming optimizer (Problem Statement, Sections 05.2-05.3, 09).

The 24-hour schedule is solved as a true LP with scipy.optimize.linprog on the
HiGHS backend -- not a heuristic, so the returned schedule is cost-optimal for
the constraint set it is given. It solves in milliseconds on one CPU core,
which is what lets this run on a free CPU-only host inside the p95 budget.

Decision variables (96 total, 24 per group):
    grid[h]        grid energy purchased in hour h
    solar_used[h]  solar energy actually used in hour h
    charge[h]      battery charge amount in hour h
    discharge[h]   battery discharge amount in hour h

Objective:   minimize  sum(grid[h] * tariff[h])

Constraints:
    - hourly energy balance                              (equality)
    - end-of-day battery neutrality                      (equality)
    - battery state-of-charge bounds every hour, honoring
      any active minimum_battery_reserve                 (inequality, prefix sums)
    - per-hour variable bounds encode effective solar after solar_reduction,
      charge/discharge rate limits, no_charge_window and no_discharge_window
      (bound forced to [0, 0]), and max_grid_window.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from gridwise.constants import (
    ACTION_CHARGE,
    ACTION_DISCHARGE,
    ACTION_IDLE,
    MAX_GRID_WINDOW,
    MINIMUM_BATTERY_RESERVE,
    NO_CHARGE_WINDOW,
    NO_DISCHARGE_WINDOW,
    NUM_HOURS,
    OUTPUT_DECIMALS,
    SOLAR_REDUCTION,
)

# Below this, an LP value is treated as exactly zero when deciding whether an
# hour charges, discharges, or idles.
_ACTION_EPSILON = 1e-6


class InfeasibleScenarioError(Exception):
    """No schedule satisfies every constraint simultaneously."""


class ScenarioConditions:
    """Per-hour numbers the LP needs, after directives have been applied."""

    def __init__(self, hours: List[Dict[str, Any]], battery: Dict[str, Any]):
        self.demand = [0.0] * NUM_HOURS
        self.tariff = [0.0] * NUM_HOURS
        self.effective_solar = [0.0] * NUM_HOURS
        for entry in hours:
            h = entry["hour"]
            self.demand[h] = float(entry["demand_kwh"])
            self.tariff[h] = float(entry["tariff_bdt_per_kwh"])
            self.effective_solar[h] = float(entry["solar_kwh"])

        base_minimum = float(battery["minimum_energy_kwh"])
        self.reserve = [base_minimum] * NUM_HOURS
        self.charge_blocked = [False] * NUM_HOURS
        self.discharge_blocked = [False] * NUM_HOURS
        self.max_grid: List[Optional[float]] = [None] * NUM_HOURS

        self.capacity = float(battery["capacity_kwh"])
        self.initial_energy = float(battery["initial_energy_kwh"])
        self.max_charge_rate = float(battery["max_charge_kwh_per_hour"])
        self.max_discharge_rate = float(battery["max_discharge_kwh_per_hour"])


def apply_directives(
    hours: List[Dict[str, Any]],
    battery: Dict[str, Any],
    directive_interpretation: List[Dict[str, Any]],
) -> ScenarioConditions:
    """Applies validated directives to the scenario (Section 05.3).

    Overlapping directives compose conservatively: the tightest reserve and the
    lowest grid cap win, and solar factors multiply.
    """
    conditions = ScenarioConditions(hours, battery)

    for entry in directive_interpretation:
        if not entry.get("applies"):
            continue
        directive_type = entry["directive_type"]
        adjustment = entry.get("structured_adjustment") or {}

        for hour in adjustment.get("hours", []):
            if directive_type == SOLAR_REDUCTION:
                conditions.effective_solar[hour] *= float(adjustment["factor"])
            elif directive_type == MINIMUM_BATTERY_RESERVE:
                conditions.reserve[hour] = max(
                    conditions.reserve[hour], float(adjustment["minimum_energy_kwh"])
                )
            elif directive_type == NO_CHARGE_WINDOW:
                conditions.charge_blocked[hour] = True
            elif directive_type == NO_DISCHARGE_WINDOW:
                conditions.discharge_blocked[hour] = True
            elif directive_type == MAX_GRID_WINDOW:
                cap = float(adjustment["max_grid_kwh"])
                current = conditions.max_grid[hour]
                conditions.max_grid[hour] = cap if current is None else min(current, cap)

    return conditions


def _build_program(conditions: ScenarioConditions) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                            np.ndarray, np.ndarray, list]:
    n = NUM_HOURS
    grid_i, solar_i, charge_i, discharge_i = 0, n, 2 * n, 3 * n
    num_vars = 4 * n

    # Objective: cost of grid energy only.
    objective = np.zeros(num_vars)
    for h in range(n):
        objective[grid_i + h] = conditions.tariff[h]

    # Bounds carry the per-hour directives that are simple caps.
    bounds: list = [(0.0, None)] * num_vars
    for h in range(n):
        cap = conditions.max_grid[h]
        bounds[grid_i + h] = (0.0, cap) if cap is not None else (0.0, None)
        bounds[solar_i + h] = (0.0, max(conditions.effective_solar[h], 0.0))
        bounds[charge_i + h] = (0.0, 0.0) if conditions.charge_blocked[h] \
            else (0.0, conditions.max_charge_rate)
        bounds[discharge_i + h] = (0.0, 0.0) if conditions.discharge_blocked[h] \
            else (0.0, conditions.max_discharge_rate)

    # Equalities: hourly balance, then end-of-day neutrality.
    a_eq = np.zeros((n + 1, num_vars))
    b_eq = np.zeros(n + 1)
    for h in range(n):
        a_eq[h, grid_i + h] = 1.0
        a_eq[h, solar_i + h] = 1.0
        a_eq[h, discharge_i + h] = 1.0
        a_eq[h, charge_i + h] = -1.0
        b_eq[h] = conditions.demand[h]
    for h in range(n):
        a_eq[n, charge_i + h] = 1.0
        a_eq[n, discharge_i + h] = -1.0
    b_eq[n] = 0.0

    # Inequalities: state of charge stays within [reserve[h], capacity] at the
    # end of every hour, expressed as prefix sums over charge/discharge.
    a_ub = np.zeros((2 * n, num_vars))
    b_ub = np.zeros(2 * n)
    for h in range(n):
        for k in range(h + 1):
            a_ub[2 * h, charge_i + k] = 1.0        # E[h] <= capacity
            a_ub[2 * h, discharge_i + k] = -1.0
            a_ub[2 * h + 1, charge_i + k] = -1.0   # E[h] >= reserve[h]
            a_ub[2 * h + 1, discharge_i + k] = 1.0
        b_ub[2 * h] = conditions.capacity - conditions.initial_energy
        b_ub[2 * h + 1] = conditions.initial_energy - conditions.reserve[h]

    return objective, a_ub, b_ub, a_eq, b_eq, bounds


def _build_plan(solution: np.ndarray, conditions: ScenarioConditions) -> List[Dict[str, Any]]:
    """Turns the raw LP solution into the published hourly_plan.

    Every published number is derived from values that have already been
    rounded, so the energy balance and the battery state transition hold
    exactly on the numbers the judge actually reads -- rather than only on
    the unrounded values the solver saw.
    """
    n = NUM_HOURS
    grid_i, solar_i, charge_i, discharge_i = 0, n, 2 * n, 3 * n

    plan: List[Dict[str, Any]] = []
    battery_energy = round(conditions.initial_energy, OUTPUT_DECIMALS)

    for h in range(n):
        charge = max(0.0, float(solution[charge_i + h]))
        discharge = max(0.0, float(solution[discharge_i + h]))

        # An hour is charge, discharge, or idle -- never both -- so net the two
        # before publishing. The LP has no incentive to do both at once, but
        # netting makes that structurally impossible in the response.
        net = charge - discharge
        if net > _ACTION_EPSILON:
            action, magnitude = ACTION_CHARGE, net
        elif net < -_ACTION_EPSILON:
            action, magnitude = ACTION_DISCHARGE, -net
        else:
            action, magnitude = ACTION_IDLE, 0.0

        battery_kwh = round(magnitude, OUTPUT_DECIMALS)

        solar_used = min(max(0.0, float(solution[solar_i + h])), conditions.effective_solar[h])
        solar_used = round(solar_used, OUTPUT_DECIMALS)
        # Rounding must never push usage above the effective solar ceiling.
        if solar_used > conditions.effective_solar[h]:
            solar_used = round(conditions.effective_solar[h], OUTPUT_DECIMALS)

        charged = battery_kwh if action == ACTION_CHARGE else 0.0
        discharged = battery_kwh if action == ACTION_DISCHARGE else 0.0

        # Derive grid from the balance equation on the rounded values, so
        # Section 9.5 holds exactly for the numbers we publish.
        grid_kwh = round(
            conditions.demand[h] + charged - solar_used - discharged, OUTPUT_DECIMALS
        )
        grid_kwh = max(0.0, grid_kwh)

        battery_energy = round(battery_energy + charged - discharged, OUTPUT_DECIMALS)

        plan.append({
            "hour": h,
            "grid_kwh": grid_kwh,
            "solar_used_kwh": solar_used,
            "battery_action": action,
            "battery_kwh": battery_kwh,
            "battery_energy_after_kwh": battery_energy,
        })

    return plan


def summarize(plan: List[Dict[str, Any]], conditions: ScenarioConditions) -> Dict[str, Any]:
    """Computes the reported totals from the published plan, not the solver.

    Section 11.3 recalculates these from hourly_plan, so they are calculated
    the same way here and cannot drift apart.
    """
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0
    for entry in plan:
        grid = entry["grid_kwh"]
        total_grid += grid
        total_cost += grid * conditions.tariff[entry["hour"]]
        peak_grid = max(peak_grid, grid)

    return {
        "total_grid_kwh": round(total_grid, OUTPUT_DECIMALS),
        "total_cost_bdt": round(total_cost, OUTPUT_DECIMALS),
        "peak_grid_kwh": round(peak_grid, OUTPUT_DECIMALS),
    }


def solve(
    hours: List[Dict[str, Any]],
    battery: Dict[str, Any],
    directive_interpretation: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Solves the scenario and returns the plan plus its recalculated totals.

    Raises InfeasibleScenarioError when the directive combination admits no
    valid schedule. Organizer scoring scenarios are guaranteed feasible, so
    this surfaces a real contradiction instead of inventing a plan.
    """
    conditions = apply_directives(hours, battery, directive_interpretation)
    objective, a_ub, b_ub, a_eq, b_eq, bounds = _build_program(conditions)

    result = linprog(
        objective,
        A_ub=a_ub, b_ub=b_ub,
        A_eq=a_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise InfeasibleScenarioError(
            "no feasible 24-hour schedule exists for this scenario and directive "
            f"combination (solver status: {result.message})"
        )

    plan = _build_plan(result.x, conditions)
    totals = summarize(plan, conditions)

    return {"hourly_plan": plan, "conditions": conditions, **totals}
