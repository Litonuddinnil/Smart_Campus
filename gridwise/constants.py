"""Domain constants fixed by the Problem Statement.

These are specification values, not tunables. They live apart from config.py
(which reads the environment) so that nothing deployment-related can change
the meaning of the challenge contract.
"""

# Planning horizon: hours 0..23 inclusive (Problem Statement, Section 07).
NUM_HOURS = 24

# Section 04.1 — the only directive types the service may ever emit.
SOLAR_REDUCTION = "solar_reduction"
MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
NO_CHARGE_WINDOW = "no_charge_window"
NO_DISCHARGE_WINDOW = "no_discharge_window"
MAX_GRID_WINDOW = "max_grid_window"
NO_OP = "no_op"

ALLOWED_DIRECTIVE_TYPES = frozenset({
    SOLAR_REDUCTION,
    MINIMUM_BATTERY_RESERVE,
    NO_CHARGE_WINDOW,
    NO_DISCHARGE_WINDOW,
    MAX_GRID_WINDOW,
    NO_OP,
})

# Section 10.3 — battery_action enum.
ACTION_CHARGE = "charge"
ACTION_DISCHARGE = "discharge"
ACTION_IDLE = "idle"
ALLOWED_BATTERY_ACTIONS = frozenset({ACTION_CHARGE, ACTION_DISCHARGE, ACTION_IDLE})

# Section 11.5 — the judge treats values within this absolute tolerance as equal.
NUMERIC_TOLERANCE = 0.01

# We round published plan values to this many decimals. Well inside the 0.01
# tolerance, and it keeps the JSON response readable.
OUTPUT_DECIMALS = 4
