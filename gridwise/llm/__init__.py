"""Language-model interpretation layer.

This package owns the only step in the pipeline that is allowed to be
non-deterministic. Everything it returns is untrusted and must pass through
gridwise.core.guardrails before the optimizer sees it.
"""
from gridwise.llm.interpreter import LLMError, interpret_notes

__all__ = ["LLMError", "interpret_notes"]
