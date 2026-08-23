"""Optional model reranker — not part of the M35 canonical path.

The required V09 reranker is the deterministic lexical scorer in
`retrieval_eval.py` (`lex-overlap-v1`). This module exists so a learner
with a local cross-encoder can later wrap the same candidate-set
interface. It is never imported by the notebook, required tests, or
`retrieval_eval.py`.

Sentence-Transformers, model-hub downloads, and paid APIs are not CI
dependencies. Approximate indexes remain M36.
"""

from __future__ import annotations


class OptionalModelRerankerUnavailable(RuntimeError):
    """Raised when the optional adapter is requested on the required path."""


def open_optional_model_reranker(*_args, **_kwargs):
    raise OptionalModelRerankerUnavailable(
        "M35 canonical reranking is lex-overlap-v1 (local, deterministic). "
        "A model reranker is not a CI or notebook dependency."
    )
