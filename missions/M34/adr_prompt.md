# M34 ADR Prompt — V09 Grounding Contract

Use `templates/ADR.md`. The decision is not pre-selected. M34 needs a
policy for V09 teaching RAG: when evidence must be included, how
citations map to spans, when to abstain, what traces are retained,
and whether unsupported claims block release.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Evidence inclusion:** [UNFILLED BY LEARNER]
- **Citation mapping:** [UNFILLED BY LEARNER]
- **Abstention policy:** [UNFILLED BY LEARNER]
- **Trace retention:** [UNFILLED BY LEARNER]
- **Unsupported-claim release gate:** [UNFILLED BY LEARNER]
- **Logging:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V09 default for teaching grounding (extractive local
synthesis, fail-closed unsupported citations, abstain when the pack
cannot support a claim). Do **not** claim the choice is a production
LLM or a vector database.

## Alternatives that must be compared

1. Canonicalize `extractive-span-v1` over the M33 exact retriever
   with rank-ordered packing, citation-to-span mapping, abstention on
   insufficient evidence, traces that retain index identity, and
   release-blocking unsupported claims.
2. Always emit a fluent top-1 answer even when the cited span does
   not support the claim; treat support checks as telemetry.
3. Make a paid live chat model the required synthesizer for CI and
   the notebook, including sampling controls.

## Evidence required

Use the with/without retrieval pair, top-k and budget observations,
unanswerable abstention despite a high distractor score, the ticket
retrieval miss versus naive top-1, the repaired unsupported-citation
failure, and the frozen holdout pass. Do not use a downloaded
generator or a required live API as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: embedding/index identity change,
eval-set change, a need for reranking (M35), approximate search
(M36), or any path that answers from parametric memory without a
support check.
