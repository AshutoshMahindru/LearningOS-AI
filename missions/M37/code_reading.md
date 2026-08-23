# Code reading — register, schema, parse, validate, execute, retry, trace

Read `run_tool_call`, `validate_arguments`, `execute_tool`,
`repair_run`, and `optional_live_propose` in
`missions/M37/tool_runtime.py`. M37's code-reading target is the
**tool-call wrapper**:

1. A produced M32 `InferenceConfig` is attached (`training_time=False`,
   `weights_updated=False`)
2. A model-call fixture is parsed; malformed JSON is a schema/parse
   failure
3. The selected tool is looked up in the registry, or the call is
   no-tool
4. Strict schema validation runs **before** `execute_tool`
5. Side-effecting tools check approval, then the idempotency store
6. The handler runs only if those gates pass
7. Results are structured (`status`, `error_kind`, `output`)
8. Repair retries are bounded; live adapters fail closed

Before running the code-reading cell, predict:

- whether a wrong-type VAT proposal reaches `execute_tool`
- which error type a valid-but-unknown SKU produces
- what `repair_run` reuses from the broken object (proposal /
  initial ledger snapshot vs module defaults)
- what the ledger `effect_count` is after an approved replay of the
  same idempotency key

Do **not** look for a LangGraph `StateGraph`, a RAG pack, a Qdrant
client, or a temperature sampler. Those are later or parallel missions.
If a failure can be diagnosed from `validation.ok` or
`ledger.effect_count`, stay at that boundary.

Do not print substring membership of a later helper (`"idempotency" in
handler source`). Probe the live objects: validation error types,
effect count, retry budget remaining.
