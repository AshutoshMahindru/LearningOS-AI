# M36 ADR Prompt — V09 Retrieval Infrastructure

Use `templates/ADR.md`. The decision is not pre-selected. M36 needs a
V09 policy for teaching retrieval infrastructure: exact versus
approximate search, local teaching store versus a remote vector
database, dense versus sparse versus hybrid, where filters run, which
fusion method is declared, how lifecycle/staleness is handled, and
what would force a migration.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Exact vs approximate:** [UNFILLED BY LEARNER]
- **Local store vs remote vector database:** [UNFILLED BY LEARNER]
- **Dense / sparse / hybrid default:** [UNFILLED BY LEARNER]
- **Filter placement:** [UNFILLED BY LEARNER]
- **Fusion method:** [UNFILLED BY LEARNER]
- **Lifecycle / staleness:** [UNFILLED BY LEARNER]
- **Recall target:** [UNFILLED BY LEARNER]
- **Latency / memory proxy:** [UNFILLED BY LEARNER]
- **Rollback / revisit conditions:** [UNFILLED BY LEARNER]
- **Logging:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V09 default for teaching retrieval infrastructure (M35 exact
oracle, local graph effort knob, payload pre-filters, BM25 sparse,
declared RRF, fail-closed dirty indexes). Do **not** claim the choice
is a production Qdrant cluster or a downloaded ANN library.

## Alternatives that must be compared

1. Keep exact in-memory cosine as the required path at this corpus
   size, use the teaching graph only to measure effort, pre-filter
   payloads, and fuse dense+sparse with RRF when lexical ids matter.
2. Treat approximate search as correct by default, filter only after
   a small top-k, and add cosine to BM25.
3. Make a required live Qdrant/FAISS client the CI path for this
   mission.

## Evidence required

Use the M35 baseline identity, `rag-ceo` ef=1 versus ef=4 neighbor
recall, the `Please reset` FilterTrace, ticket/invoice/password
channel ids, mix-versus-RRF on password, and the dirty-insert
StoreStaleError. Do not use a required live cluster as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: corpus size where exact scan is
too slow, a need for a managed vector database, eval-set change,
embedding identity change, filter cardinality that breaks graph
connectivity, or any path that mixes raw score scales.
