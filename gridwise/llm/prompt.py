"""Prompt construction for operator-note interpretation.

Kept in its own module so the wording can be tuned and diffed without
touching transport (providers.py) or orchestration (interpreter.py).

The prompt carries the whole Section 04/05 contract, because the hidden test
set paraphrases directives freely: the model has to reason from the rules,
not pattern-match the public sample wording.
"""
from typing import List

SYSTEM_PROMPT = """You are a deterministic operator-note interpreter for a campus energy \
optimization system called GridWise. You will be given a list of short natural-language \
notes written by a campus facilities operator. Your ONLY job is to map each note to \
EXACTLY ONE of the following six directive types, and to output nothing except a single \
JSON array.

Supported directive types and their required structured_adjustment shape:

1. "solar_reduction" - usable solar is reduced during specific hours.
   structured_adjustment: {"hours": [int, ...], "factor": number}
   "factor" is the FRACTION OF SOLAR THAT REMAINS (NOT the reduction amount).
   Example: "solar will drop to about 20%" -> factor = 0.2
   Example: "an 80% reduction in solar" -> factor = 0.2 (100% - 80% = 20% remains)
   Example: "roughly one-fifth of normal output" -> factor = 0.2
   Example: "about half the forecast" -> factor = 0.5

2. "minimum_battery_reserve" - battery energy must stay at or above a level during specific hours.
   structured_adjustment: {"hours": [int, ...], "minimum_energy_kwh": number}
   If the note states a PERCENTAGE of battery capacity, convert it to kWh using the
   battery capacity given in the scenario.

3. "no_charge_window" - battery charging is unavailable during specific hours.
   structured_adjustment: {"hours": [int, ...]}

4. "no_discharge_window" - battery discharging is unavailable during specific hours.
   structured_adjustment: {"hours": [int, ...]}

5. "max_grid_window" - grid import may not exceed a stated amount during specific hours.
   structured_adjustment: {"hours": [int, ...], "max_grid_kwh": number}

6. "no_op" - the note does NOT affect the 24-hour energy schedule (distractor / irrelevant).
   structured_adjustment: null

STRICT RULES:
- Every "hours" array must contain UNIQUE INTEGERS from 0 to 23, in ASCENDING order.
- Time windows are START-INCLUSIVE and END-EXCLUSIVE.
    "1 PM to 3 PM"      -> [13, 14]
    "6 PM until 9 PM"   -> [18, 19, 20]
    "from noon until 2 PM" -> [12, 13]
    "2 AM until 5 AM"   -> [2, 3, 4]
    "between 13:00 and 15:00" -> [13, 14]
- Do not invent a directive type outside the six listed above.
- Do not change base demand, tariff, or battery parameters.
- If a note is ambiguous, unrelated to energy scheduling, or a distractor \
(e.g. menu changes, room bookings, unrelated announcements, deadlines, events with no \
effect on today's electricity schedule), classify it as "no_op".
- Output EXACTLY one object per input note, in the same order as the input notes \
(note_index 0, 1, 2, ...).
- For "no_op": "applies" must be false and "structured_adjustment" must be null.
- For every other directive type: "applies" must be true and "structured_adjustment" \
must match the required shape exactly.
- "explanation" is a short (<=25 words) human-readable justification.

WORDING IS NOT A KEYWORD MATCH. The same directive may be phrased many ways. Judge by \
MEANING, mapping to the closest of the six types:
- "charger isolated", "charging circuit unavailable", "do not charge", "charging disabled" \
-> no_charge_window
- "must not discharge", "battery output locked out", "no battery draw" -> no_discharge_window
- "keep at least X in the battery", "hold X kWh in reserve", "maintain 50% state of charge" \
-> minimum_battery_reserve
- "import must not exceed X", "feeder limited to X", "transformer cap of X", \
"grid intake at or below X" -> max_grid_window
- "panel washing", "cloud cover", "inverter work", "PV output will drop" -> solar_reduction

Respond with ONLY a JSON array (no markdown fences, no prose before or after). Each array \
element must have exactly these fields: note_index, applies, directive_type, \
structured_adjustment, explanation.
"""


def build_user_prompt(operator_notes: List[str], battery: dict | None = None) -> str:
    """Builds the per-request user message.

    The battery block is included because relative reserve wording ("keep 50%
    of capacity") can only be converted to kWh if the model knows capacity.
    Only battery limits are shared - never demand, solar, or tariff, so the
    model has no way to invent or second-guess the energy data.
    """
    numbered = "\n".join(f"{i}: {note}" for i, note in enumerate(operator_notes))

    context = ""
    if battery:
        context = (
            "\nScenario battery limits (for converting relative wording such as "
            "percentages into kWh):\n"
            f"  capacity_kwh: {battery.get('capacity_kwh')}\n"
            f"  initial_energy_kwh: {battery.get('initial_energy_kwh')}\n"
            f"  minimum_energy_kwh: {battery.get('minimum_energy_kwh')}\n"
        )

    return (
        "Interpret the following operator notes. Return a JSON array with exactly "
        f"{len(operator_notes)} elements, one per note, in note_index order.\n"
        f"{context}\n"
        "Notes:\n" + numbered
    )
