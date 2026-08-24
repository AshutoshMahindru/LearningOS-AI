"""Optional LLM-as-judge adapter — not part of the M40 canonical path.

The required V11 graders are the deterministic checkers in
`evaluation_harness.py`. This module exists so a learner with a local
or licensed model can later wrap a rubric. It is never imported by the
notebook, required tests, or `evaluation_harness.py` as a sole grader.

Paid eval SDKs, hosted judges, and vendor credentials are not CI
dependencies. Named source: anthropic-evals.
"""

from __future__ import annotations


class OptionalLLMJudgeUnavailable(RuntimeError):
    """Raised when the optional adapter is requested on the required path."""


def open_optional_llm_judge(*_args, **_kwargs):
    raise OptionalLLMJudgeUnavailable(
        "M40 canonical graders are deterministic (schema, citation, "
        "termination, idempotency). An LLM-as-judge adapter is not a CI "
        "or notebook dependency and cannot be the sole required grader."
    )
