# No-AI gate — design memory, routes, and a bounded fallback

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Do not reuse notebook outputs. Use only the numbers and traces below.

## Fixture (fresh)

A counter desk, not the notebook SKU-7 purchase:

```
sku                = SKU-21
stale remembered   = 13.0
written_at         = 5
expires_at         = 8
now                = 40
current catalog    = 27.0
irrelevant         = BIN-8 qty 4
max_attempts       = 2
circuit_threshold  = 2
```

Classify each as workflow state, persistent memory, or neither:

- the current `node` of an M38 run
- `SKU-21` price `13.0` with provenance and expiry
- a pending `post_ledger_entry` action
- the haiku a coworker recited at lunch

Routes must be chosen from explicit conditions, not vibes:

```
1. Refund order 4401 for SKU-21.
2. Look up the catalog price of SKU-21.
3. Schedule a picnic on the loading dock.
```

Allowed route names (use only these, plus `no_match`):

```
catalog_purchase, catalog_lookup, no_match
```

A teammate says: "the expired `13.0` is close enough; post it and
mark complete." Their trace:

```
now=40 expires_at=8 used_memory_ids=[mem-old]
posted_amount=13.0 terminal=complete degraded=False
```

Another teammate's ladder with both rungs failing:

```
attempt 1 primary fail
attempt 2 lookup_only fail
attempt 3 primary fail
...
attempt 11 lookup_only fail
circuit_open=False
```

## Part A: classify

Label the four sample items as workflow state, persistent memory, or
neither. One short list.

## Part B: provenance / expiry

Name the fields you would store on a persistent price so a reader can
tell who wrote it, when, and when it dies. One short list.

## Part C: routes

Choose a route for each of the three fresh cases from explicit
conditions. Include the no-match case.

## Part D: two-level fallback with a hard stop

Design primary plus one fallback for a purchase of SKU-21, with
`max_attempts=2` and a circuit that opens after two consecutive
failures. What is the terminal if both rungs fail?

## Part E: diagnose

Using one of the teammate traces, diagnose either the stale-memory
complete or the unbounded fallback loop. One short paragraph. Do not
"prompt the model to remember the expiry."

Pass requires a classification list, provenance fields, three route
choices, a bounded fallback with a hard stop, and a diagnosis, plus
an oral defense.
Leave all learner responses unfilled in the repository.
