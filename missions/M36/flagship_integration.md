# V09 integration — close the grounded knowledge system

## M35 → M36 boundary

M35 already measured ranking and handed a candidate/rerank contract.
M36 is the last V09 mission: **choose retrieval infrastructure**
against that oracle.

```
frozen labels → M35 exact oracle
  → teaching graph / ef
  → payload filters
  → sparse + declared RRF
  → lifecycle
  → V09 store ADR
```

The observable V09 surface after M36 is a defended choice: exact
versus approximate at teaching scale, where filters run, how dense
and sparse combine, and when the index must be rebuilt. The
measurable retrieval component for M40 is the same identity-bearing
candidate lists (`chunk_id`, ranks, fusion method, store/index ids).

## What M36 must not change

M36 does not require Qdrant or FAISS in CI. It does not open
temperature / top-p sampling (M32). It does not open tool calling
(M37) or a stateful agent (M38). It does not edit `datasets/M34`
labels. It does not treat `sentence-transformers` as a required
download.

## V09 close / M40 handoff

V09 **does not close** because this package was implemented.
Learner evidence, the infrastructure ADR, and formal review are
still required. After those, V09 hands M40 a retrieval component
that can be measured: frozen eval version, exact oracle identity,
optional approximate effort, declared fusion, and filter/lifecycle
invariants.

Reusable artifacts: `open_teaching_store`, `exact_search` /
`approximate_search` / `sparse_search` / `hybrid_search`,
`HybridHit.as_evidence`, `InfraConfig.identity`, frozen M34 labels,
and `datasets/M36/expected.json` fixture properties (not learner
evidence).
