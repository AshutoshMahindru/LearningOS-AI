# M35 ADR Prompt — V09 Chunking / Candidate / Rerank Policy

Use `templates/ADR.md`. The decision is not pre-selected. M35 needs a
policy for V09 teaching retrieval: default chunking, candidate k,
when to rerank, which metrics govern, what latency proxy is honest,
how eval leakage is prevented, and when to roll back.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Chunking policy:** [UNFILLED BY LEARNER]
- **Candidate-k policy:** [UNFILLED BY LEARNER]
- **Rerank policy:** [UNFILLED BY LEARNER]
- **Metric set:** [UNFILLED BY LEARNER]
- **Latency budget:** [UNFILLED BY LEARNER]
- **Eval governance:** [UNFILLED BY LEARNER]
- **Rollback / revisit conditions:** [UNFILLED BY LEARNER]
- **Logging:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V09 default for teaching retrieval quality (frozen M34 eval,
versioned chunks, exact candidates, local lexical rerank, slice
metrics, fail-closed eval leakage). Do **not** claim the choice is a
production vector database or a downloaded cross-encoder.

## Alternatives that must be compared

1. Canonicalize `lex-overlap-v1` over exact M33 candidates at k=3 on
   the frozen sentence chunks, report aggregate **and** critical-slice
   nDCG/MRR, and treat eval leakage / post-hoc relabeling as release
   blockers.
2. Always merge documents into one chunk and report only average
   recall@k, retuning labels after each run.
3. Make a required Sentence-Transformers cross-encoder plus Qdrant/HNSW
   the CI path for this mission.

## Evidence required

Use the frozen-label hash, baseline versus lex rerank on identical
candidates, k=1 versus k=5 candidate recall, merged versus windowed
chunk versions, hard-negative degradation, aggregate versus critical
slice, and the repaired leakage run. Do not use a downloaded reranker
or a required live API as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: eval-set change, embedding/index
identity change, a need for approximate search (M36), a model
reranker with a documented local path, or any path that edits labels
after seeing ranks.
