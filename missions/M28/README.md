# M28 — Search Meaning with Embeddings

Token IDs from M27 are not meanings. The useful whole here is a **small
retrieval**:

`text → (bundled) vector → cosine against a corpus → ranked neighbors`

Every vector carries a **provenance contract**: model, version, width,
metric, normalization, and pooling. Mixing stores without checking that
contract can still emit a plausible ranking. That is the failure, not a
reason to anthropomorphize the numbers.

This mission uses a **bundled** teaching encoder (`v06-teaching-meanpool`,
version `v06.1`): frozen token semantics, mean pooling, L2, cosine.
Nothing is downloaded. The same texts also have a lexical Jaccard
baseline so overlap and geometry can disagree.

This mission does **not** re-teach tokenization (M27), does **not**
weight vectors by context (M29), and does **not** build a search service
or index (M33). Inspect nearest neighbors on a tiny corpus first.

Canonical sources: `sentence-transformers` and `hf-llm-course` via
`data/source_registry.json`. Use the bundled fixtures; do not download
encoder weights.

Implementation status is not learner completion. Predictions, no-AI
work, ADR decisions, and competence remain intentionally unfilled.
