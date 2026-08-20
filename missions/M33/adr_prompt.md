# M33 ADR Prompt — V08 Index Identity and Rebuild Policy

Use `templates/ADR.md`. The decision is not pre-selected. M33 needs a
policy for the V08 teaching retriever: which index identity is
canonical, how embedding and corpus versions are bound, how filters and
ties work, and what happens when the index is stale or mixed.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Embedding model and version:** [UNFILLED BY LEARNER]
- **Corpus version / hash:** [UNFILLED BY LEARNER]
- **Similarity metric:** [UNFILLED BY LEARNER]
- **Filter schema:** [UNFILLED BY LEARNER]
- **Tie break:** [UNFILLED BY LEARNER]
- **Stale / mixed index policy:** [UNFILLED BY LEARNER]
- **Logging:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V08 default for teaching retrieval (exact in-memory index
identity, fail-closed freshness, rebuild versus serve-stale). Do **not**
claim the choice is a production vector database.

## Alternatives that must be compared

1. Canonicalize `v08-exact-memory` over `v06-teaching-meanpool` v06.1
   with cosine/L2, declared filter schema, `(-score, chunk_id)` ties,
   and fail-closed rebuild-or-reject on source-hash or fingerprint
   mismatch.
2. Serve stale hits when cosine still looks high and rebuild later in
   batch; treat corpus hash as telemetry only.
3. Make a local Qdrant collection the required backend for CI and the
   notebook, including approximate search.

## Evidence required

Use the exact-baseline parity, filter eligibility, labeled-versus-score
hard cases, latency/`scored_candidates` observations, the repaired
stale-index failure, and the incompatible `v06.2` gate. Do not use a
downloaded encoder or a required Qdrant service as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: embedding model/version/width
change, corpus hash change, metric change, filter-schema change, a
need for approximate search (M36), or any pipeline that generates
answers from hits (M34) without keeping this retriever as a boundary.
