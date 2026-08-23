"""Optional LangGraph adapter — not part of the M38 canonical path.

The required V10 workflow is the local deterministic state machine in
`agent_workflow.py`. This module exists so a learner with a licensed
LangGraph install can later wrap the same state schema. It is never
imported by the notebook, required tests, or `agent_workflow.py`.

Paid APIs, hosted checkpointers, and LangGraph/Anthropic SDKs are not
CI dependencies. Persistent memory, routing, and fallbacks remain M39.
"""

from __future__ import annotations


class OptionalLangGraphUnavailable(RuntimeError):
    """Raised when the optional adapter is requested on the required path."""


def compile_optional_graph(*_args, **_kwargs):
    raise OptionalLangGraphUnavailable(
        "M38 canonical path is the local deterministic state machine. "
        "A LangGraph adapter is not a CI or notebook dependency."
    )
