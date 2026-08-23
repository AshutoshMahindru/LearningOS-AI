# No-AI gate — defend a tool contract from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Do not reuse notebook outputs. Use only the numbers and calls below.

## Fixture (fresh)

A duty tool, not the notebook VAT tool:

```
compute_duty(value, rate) -> duty = value * rate, total = value + duty
value = 40
rate  = 0.125
```

Rate is a fraction in `[0, 1]`, not a percent. Extra fields are
forbidden. Boolean is not a number.

Four proposed calls (exactly as a model might emit them):

```
P1: {"tool": "compute_duty", "arguments": {"value": 40, "rate": 0.125}}
P2: {"tool": "compute_duty", "arguments": {"value": 40}}
P3: {"tool": "compute_duty", "arguments": {"value": 40, "rate": 0.125, "currency": "USD"}}
P4: {"tool": "compute_duty", "arguments": {"value": "forty", "rate": 0.125}}
```

A side-effecting refund tool `issue_refund(account, amount,
idempotency_key)` posts cash. Two timeout-retry records share
`idempotency_key = "refund-40"`.

A teammate says: "the JSON parsed, so the tool is allowed to run."

## Part A: schema arithmetic

1. What are `duty` and `total` for a valid call with the fixture
   numbers? Show the multiplication.
2. Classify P1–P4 as valid or invalid. For each invalid call, name
   the issue kind (`missing`, `extra`, or `wrong_type`).

## Part B: idempotency

If the first attempt posted and the timeout retry uses the same
idempotency key, how many times must the cash effect apply? One
sentence.

## Part C: error locus

A call `{"tool": "compute_duty", "arguments": {"value": 40, "rate": 0.125}}`
is schema-valid, but `compute_duty` raises because the duty table is
offline. Is that a schema failure, an orchestration failure, or a
tool failure? One sentence.

## Part D: parseable JSON is not enough

The teammate's claim is that parseable JSON is sufficient reliability.
In two sentences, reject or accept that claim using P3 or P4.

## Part E: replay fields

List the trace fields that must be recorded to replay a side-effecting
call without posting twice.

Pass requires duty arithmetic, four-way classification, an
idempotency count, an error-locus choice, a JSON-is-not-enough
argument, and an oral defense.
Leave all learner responses unfilled in the repository.
