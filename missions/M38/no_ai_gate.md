# No-AI gate — reconstruct a bounded workflow from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Do not reuse notebook outputs. Use only the numbers and traces below.

## Fixture (fresh)

A warehouse workflow, not the notebook SKU-7 purchase:

```
lookup_bin(bin_id) -> {bin_id, qty}
reserve_stock(bin_id, qty, idempotency_key)  # side-effecting; needs approval
bin_id = BIN-4
qty    = 9
max_steps = 5
idempotency_key = reserve-bin-4
```

`reserve_stock` must not run until a human approves. Extra fields on
tool arguments are forbidden. The model fixture is not the state.

Allowed nodes (draw only these, plus edges you justify):

```
start, decide, validate, approve, execute, assimilate,
complete, denied, failed, loop_exhausted
```

A teammate says: "the chat already mentioned the lookup, so resume can
skip storing last_tool_result."

Another teammate's trace after a granted reserve:

```
node=execute pending=reserve_stock key=reserve-bin-4
effect_count=1 entry_id=1
then node rewound to execute, skip_idempotency, effect_count=2 entry_id=2
```

A third run uses a fixture that keeps proposing `lookup_bin` and never
emits `done`. Their machine has no `max_steps` check.

## Part A: draw the machine

1. Draw a fresh bounded state machine for lookup-then-approved-reserve.
   Label terminals.
2. Mark the edge that is illegal: `start -> execute`.

## Part B: resume state

Name the state fields that must be in a checkpoint to resume after the
lookup without re-looking-up and without posting twice. One short list.

## Part C: missing terminal

The unresolved-lookup run never stops. Which terminal is missing from
that machine, and which counter should trigger it?

## Part D: approval

Where does human approval belong relative to `validate` and `execute`?
One sentence. What is `effect_count` after a deny?

## Part E: replayed side effect

Using the teammate's reserve trace, diagnose why `effect_count` became
2. One short paragraph. Do not "prompt the model to remember."

Pass requires a drawn machine, a resume-field list, a missing-terminal
identification, an approval placement, and a replay diagnosis, plus an
oral defense.
Leave all learner responses unfilled in the repository.
