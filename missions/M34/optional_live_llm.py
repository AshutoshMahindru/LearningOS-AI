"""Optional live-model adapter — not part of the M34 canonical path.

The required V09 synthesizer is the local extractive copier in
`rag_pipeline.py`. This module exists so a learner with a local or
licensed model can later wrap the same pack -> answer interface. It is
never imported by the notebook, required tests, or `rag_pipeline.py`.

Paid APIs, hosted chat completions, and decoding labs are not CI
dependencies. Token sampling remains M32.
"""

from __future__ import annotations


class OptionalLiveLLMUnavailable(RuntimeError):
    """Raised when the optional adapter is requested on the required path."""


def open_optional_live_synthesizer(*_args, **_kwargs):
    raise OptionalLiveLLMUnavailable(
        "M34 canonical synthesis is extractive-span-v1 (local, deterministic). "
        "A live model adapter is not a CI or notebook dependency."
    )
