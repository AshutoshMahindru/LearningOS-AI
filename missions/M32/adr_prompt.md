# M32 ADR Prompt — V07 Adaptation Hierarchy and Inference Configuration

Use `templates/ADR.md`. The decision is not pre-selected. M32 needs a
policy for the V07 artifact M33/M34/M37 will inherit: when to change
prompt, retrieve, call tools, or change parameters, and which inference
fields must be logged.

## Unfilled decision record

- **Status:** [UNFILLED BY LEARNER]
- **Date:** [UNFILLED BY LEARNER]
- **Owner:** [UNFILLED BY LEARNER]
- **Decision:** [UNFILLED BY LEARNER]
- **Adaptation hierarchy (prompt / retrieval / tools / parameters):** [UNFILLED BY LEARNER]
- **When fine-tuning is forbidden as the first lever:** [UNFILLED BY LEARNER]
- **Required inference logs:** [UNFILLED BY LEARNER]
- **Seed / temperature / stop policy:** [UNFILLED BY LEARNER]
- **Live-provider fallback:** [UNFILLED BY LEARNER]
- **Teaching-scale limits:** [UNFILLED BY LEARNER]
- **Evidence:** [UNFILLED BY LEARNER]
- **Alternatives considered:** [UNFILLED BY LEARNER]
- **Trade-offs:** [UNFILLED BY LEARNER]
- **Migration / revisit triggers:** [UNFILLED BY LEARNER]

## Decision to make

Choose a V07 default for adaptation order and inference-provider
configuration (what must be recorded on every completion, when two
outputs may be compared as a model change, which lever is first for
freshness versus behavior). Do **not** claim the 4-token fixture is a
production decoder, and do not implement M33 search, M34 RAG, or M37
tool execution in this ADR.

Parameter change may name instruction-tuning, LoRA, QLoRA, or full
fine-tune as options. Naming is not an implementation.

## Alternatives that must be compared

1. Ordered hierarchy prompt → retrieval → tools → parameters; log
   checkpoint id, prompt, temperature, top-k/p, seed, stop, max-tokens,
   and `do_sample` on every call; refuse to treat temperature as quality.
2. Fine-tune first whenever answers look wrong; log only the output
   string so traces stay small.
3. Treat temperature/top-p as the adaptation lever and skip retrieval
   or tools for freshness and computation problems.

## Evidence required

Use the greedy/temperature tables, the seed replay, the stop-budget
traces, the two-prompt continuations, the uncontrolled-settings
failure, and the stale-hours adaptation repair. Do not use a vendor
model card as evidence.

## Monitoring and revisit conditions

Specify what would force a revisit: a real provider replacing the local
score table, an instruction-tuning stage, attaching an M33 retriever,
an M34 context pack, an M37 tool loop, or any comparison of two
completions that omits seed, temperature, prompt, or stop.
