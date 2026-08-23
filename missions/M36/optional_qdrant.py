"""Optional live vector-store adapter — not part of the M36 canonical path.

The required V09 store is the local teaching adapter in
`hybrid_retrieval.py` (exact cosine oracle, graph effort knob, BM25,
declared RRF). This module exists so a learner with a *local* Qdrant
process can later wrap the same search interface. It is never imported
by the notebook, required tests, or `hybrid_retrieval.py`.

Managed clusters, API keys, and FAISS/Qdrant client libraries are not
CI dependencies. Approximate search effort is taught by the local
adapter, not by downloading an ANN implementation.
"""

from __future__ import annotations


class OptionalQdrantUnavailable(RuntimeError):
    """Raised when the optional adapter is requested on the required path."""


def open_optional_qdrant(*_args, **_kwargs):
    raise OptionalQdrantUnavailable(
        "M36 canonical retrieval is the local teaching store "
        "(exact oracle + graph effort + sparse + RRF). "
        "A live Qdrant client is not a CI or notebook dependency."
    )
