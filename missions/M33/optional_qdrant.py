"""Optional local Qdrant adapter — not part of the M33 canonical path.

Exact in-memory cosine search (`ExactIndex`) is the required V08 baseline.
This module exists so a learner with a *local* Qdrant process can later
explore the same retriever interface. It is never imported by the notebook,
required tests, or `semantic_search.py`.

Approximate indexes, HNSW search effort, and hybrid fusion belong to M36.
This file does not import `qdrant_client` at module load and must not
become a CI dependency.
"""

from __future__ import annotations


class OptionalQdrantUnavailable(RuntimeError):
    """Raised when the optional adapter is requested on the required path."""


def open_optional_client(*_args, **_kwargs):
    raise OptionalQdrantUnavailable(
        "M33 canonical retrieval is ExactIndex (in-memory, exact cosine). "
        "Qdrant is not a CI or notebook dependency. M36 owns required "
        "vector-database infrastructure."
    )
