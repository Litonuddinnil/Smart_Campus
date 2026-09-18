"""Deterministic core: guardrails, optimizer, and the final replay validator.

Nothing in this package calls a language model or the network. Given the same
inputs it always produces the same output, which is what makes the judge's
independent replay reproducible.
"""
