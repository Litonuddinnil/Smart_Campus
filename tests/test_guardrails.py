"""Unit tests for gridwise.core.guardrails (Problem Statement, Section 08)."""
from gridwise.core import guardrails


def test_valid_hours_list_accepts_ascending_unique():
    assert guardrails.valid_hours_list([13, 14])
    assert guardrails.valid_hours_list([0, 1, 2, 3])


def test_valid_hours_list_rejects_bad_input():
    assert not guardrails.valid_hours_list([])
    assert not guardrails.valid_hours_list([14, 13])       # not ascending
    assert not guardrails.valid_hours_list([13, 13])       # duplicate
    assert not guardrails.valid_hours_list([13, 24])       # out of range
    assert not guardrails.valid_hours_list([13, "14"])     # wrong type
    assert not guardrails.valid_hours_list("13,14")        # not a list
    assert not guardrails.valid_hours_list([True, False])  # bools are not ints here


def test_no_op_forced_regardless_of_model_fields():
    notes = ["irrelevant note"]
    raw = [{
        "note_index": 0, "applies": True, "directive_type": "no_op",
        "structured_adjustment": {"hours": [1]}, "explanation": "wrong but harmless",
    }]
    cleaned = guardrails.validate_and_clean(raw, notes)
    assert cleaned == [{
        "note_index": 0, "applies": False, "directive_type": "no_op",
        "structured_adjustment": None, "explanation": "wrong but harmless",
    }]


def test_valid_solar_reduction_passes_through():
    notes = ["Solar drops to 20% from 1pm to 3pm."]
    raw = [{
        "note_index": 0, "applies": True, "directive_type": "solar_reduction",
        "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
        "explanation": "ok",
    }]
    cleaned = guardrails.validate_and_clean(raw, notes)
    assert cleaned[0]["directive_type"] == "solar_reduction"
    assert cleaned[0]["applies"] is True
    assert cleaned[0]["structured_adjustment"] == {"hours": [13, 14], "factor": 0.2}


def test_solar_factor_out_of_range_falls_back_to_no_op():
    notes = ["Solar drops."]
    raw = [{
        "note_index": 0, "applies": True, "directive_type": "solar_reduction",
        "structured_adjustment": {"hours": [13], "factor": 1.5},
        "explanation": "bad factor",
    }]
    cleaned = guardrails.validate_and_clean(raw, notes)
    assert cleaned[0]["directive_type"] == "no_op"
    assert cleaned[0]["applies"] is False


def test_unsupported_directive_type_falls_back_to_no_op():
    notes = ["Something unusual."]
    raw = [{
        "note_index": 0, "applies": True, "directive_type": "shutdown_grid",
        "structured_adjustment": {"hours": [1]}, "explanation": "invented type",
    }]
    cleaned = guardrails.validate_and_clean(raw, notes)
    assert cleaned[0]["directive_type"] == "no_op"


def test_missing_note_gets_safe_no_op():
    notes = ["note a", "note b"]
    raw = [{
        "note_index": 0, "applies": False, "directive_type": "no_op",
        "structured_adjustment": None, "explanation": "fine",
    }]
    cleaned = guardrails.validate_and_clean(raw, notes)
    assert len(cleaned) == 2
    assert cleaned[1]["directive_type"] == "no_op"
    assert cleaned[1]["note_index"] == 1


def test_duplicate_note_index_keeps_first_only():
    notes = ["note a"]
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
         "structured_adjustment": {"hours": [1]}, "explanation": "first"},
        {"note_index": 0, "applies": True, "directive_type": "no_discharge_window",
         "structured_adjustment": {"hours": [2]}, "explanation": "duplicate, ignored"},
    ]
    cleaned = guardrails.validate_and_clean(raw, notes)
    assert len(cleaned) == 1
    assert cleaned[0]["directive_type"] == "no_charge_window"


def test_non_list_raw_output_degrades_to_all_no_op():
    notes = ["a", "b", "c"]
    cleaned = guardrails.validate_and_clean("not a list", notes)
    assert len(cleaned) == 3
    assert all(c["directive_type"] == "no_op" for c in cleaned)


def test_reserve_above_capacity_rejected():
    notes = ["Keep 500 kWh reserved."]
    raw = [{
        "note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
        "structured_adjustment": {"hours": [18], "minimum_energy_kwh": 500},
        "explanation": "too high",
    }]
    cleaned = guardrails.validate_and_clean(raw, notes, battery_capacity=200.0)
    assert cleaned[0]["directive_type"] == "no_op"


def test_safe_fallback_marks_every_note_no_op():
    notes = ["a", "b"]
    fallback = guardrails.safe_fallback(notes, "provider unreachable")
    assert len(fallback) == 2
    assert all(f["applies"] is False and f["directive_type"] == "no_op" for f in fallback)


def test_note_index_order_is_always_ascending():
    notes = ["a", "b", "c"]
    raw = [
        {"note_index": 2, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "c"},
        {"note_index": 0, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "a"},
    ]
    cleaned = guardrails.validate_and_clean(raw, notes)
    assert [c["note_index"] for c in cleaned] == [0, 1, 2]
