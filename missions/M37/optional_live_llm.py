"""Optional live-model adapter — not part of the M37 canonical path.

The required V10 proposer is the local model-call fixture table in
`tool_runtime.py`. This module exists so a learner with a local or
licensed model can later wrap the same intent -> proposal interface.
It is never imported by the notebook, required tests, or
`tool_runtime.py`.

Paid APIs, hosted chat completions, and LangGraph/Anthropic SDKs are
not CI dependencies. Token sampling remains M32.
"""

from __future__ import annotations


class OptionalLiveLLMUnavailable(RuntimeError):
    """Raised when the optional adapter is requested on the required path."""


def open_optional_live_proposer(*_args, **_kwargs):
    raise OptionalLiveLLMUnavailable(
        "M37 canonical proposals are local model-call fixtures. "
        "A live model adapter is not a CI or notebook dependency."
    )
