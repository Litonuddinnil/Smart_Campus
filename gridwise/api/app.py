"""Flask app factory for the GridWise API.

Endpoints:
    GET  /health            -> {"status": "ok"}
    POST /optimize-energy   -> operator-note interpretation + 24-hour schedule

Pipeline (Problem Statement, Section 03):
    Energy Data + Operator Notes
      -> LLM Interpreter        (gridwise.llm)
      -> Guardrail Validator    (gridwise.core.guardrails)
      -> Math Optimizer         (gridwise.core.optimizer)
      -> Final Validator        (gridwise.core.validator)
      -> API Response
"""
import logging
import traceback

from flask import Flask, jsonify, request

from gridwise import config
from gridwise.api import schema
from gridwise.core import guardrails, optimizer, validator
from gridwise.llm import LLMError, interpret_notes

logger = logging.getLogger(__name__)


def _build_plan_summary(directive_interpretation, result) -> str:
    applied = sorted({d["directive_type"] for d in directive_interpretation if d["applies"]})
    directive_text = "applying " + ", ".join(applied) if applied else "no operator directives applied"
    return (
        f"Optimized 24-hour grid-solar-battery schedule ({directive_text}); "
        f"total grid cost {result['total_cost_bdt']} BDT, "
        f"peak grid draw {result['peak_grid_kwh']} kWh."
    )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/optimize-energy")
    def optimize_energy():
        # 1. Parse JSON safely -- malformed JSON must not crash the service
        #    (Section 06.1: 400 for malformed/structurally invalid requests).
        try:
            payload = request.get_json(force=False, silent=False)
        except Exception:
            return jsonify({"error": "Malformed JSON in request body."}), 400

        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        # 2. Validate request schema (Section 07).
        try:
            schema.validate_request(payload)
        except schema.ValidationError as exc:
            return jsonify({"error": exc.message}), 400

        scenario_id = payload["scenario_id"]
        operator_notes = payload["operator_notes"]
        hours = schema.normalize_hours(payload["hours"])
        battery = payload["battery"]

        # 3. LLM interpretation -- untrusted output (Section 08).
        try:
            raw_interpretation = interpret_notes(operator_notes, battery)
        except LLMError as exc:
            logger.warning("LLM interpretation failed for %s: %s", scenario_id, exc)
            raw_interpretation = []  # guardrails below turn every note into a safe no_op

        # 4. Deterministic guardrail validation -- always runs, live or fallback.
        directive_interpretation = guardrails.validate_and_clean(
            raw_interpretation, operator_notes, battery_capacity=battery.get("capacity_kwh")
        )

        # 5. Math optimizer.
        try:
            result = optimizer.solve(hours, battery, directive_interpretation)
        except optimizer.InfeasibleScenarioError as exc:
            logger.error("Infeasible scenario %s: %s", scenario_id, exc)
            return jsonify({"error": str(exc)}), 422
        except Exception:
            logger.error(
                "Optimizer crashed for scenario %s:\n%s", scenario_id, traceback.format_exc()
            )
            return jsonify({"error": "Internal optimization error."}), 500

        # 6. Final replay validator -- independently re-checks the plan we are
        #    about to return, the same way the judge will (Section 11.2-11.3).
        #    A violation here means our own bug, not bad input; it is logged
        #    loudly rather than silently swallowed.
        violations = validator.replay_and_check(
            result["hourly_plan"], result["conditions"], directive_interpretation, battery
        )
        if violations:
            logger.error(
                "Final validator found %d violation(s) for scenario %s: %s",
                len(violations), scenario_id, violations,
            )

        plan_summary = _build_plan_summary(directive_interpretation, result)

        response = {
            "scenario_id": scenario_id,
            "directive_interpretation": directive_interpretation,
            "hourly_plan": result["hourly_plan"],
            "total_grid_kwh": result["total_grid_kwh"],
            "total_cost_bdt": result["total_cost_bdt"],
            "peak_grid_kwh": result["peak_grid_kwh"],
            "plan_summary": plan_summary,
        }
        return jsonify(response), 200

    @app.errorhandler(404)
    def not_found(_exc):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_exc):
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def internal_error(_exc):
        return jsonify({"error": "Internal server error."}), 500

    return app


def _configure_logging() -> None:
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))


_configure_logging()
