# M05 — Understand Arrays by Making Python Too Slow

## Mission objective

Experience why homogeneous arrays and vectorized operations matter by first solving a useful order-pricing problem with ordinary Python loops, then scaling the same calculation until interpreter overhead is visible and replacing the loops with NumPy.

## Whole-first route

Use the complete calculation before studying array vocabulary:

**order quantities × product prices → discounts → shipping rule → order totals**

The notebook keeps the business rule identical while changing only the computational representation:

**useful Python loop → safe scale → timing prediction → loop timing → NumPy vectorization → correctness check → timing comparison → shapes/broadcasting → transfer**

## What the learner must be able to explain

- A `shape` names the length of every axis; it is not merely the total element count.
- A `dtype` records the common representation NumPy uses for array elements and affects arithmetic behavior.
- An `axis` identifies the dimension removed by an aggregation.
- Broadcasting aligns shapes from the trailing dimensions and only combines equal dimensions or dimensions of size one.
- Indexing and slicing select positions or regions; boolean indexing selects by a vectorized condition.
- Vectorization expresses elementwise and aggregate work as array operations rather than Python-level element loops.
- A faster result is only acceptable after numerical equivalence is checked.

## Runtime boundaries

The source notebook is CPU-only, deterministic apart from ordinary timing variation, network-free, and contains no secrets or paid API calls. Its scaled workload is intentionally large enough to expose loop overhead but capped to remain safe on a laptop. Timing ratios are observations, not pass/fail assertions.

## Deliverables

Learner-produced deliverables are defined in `evidence_contract.yaml`. No learner predictions, timing results, scores, or completion claims are prefilled in this repository.

## Source policy

M05 does not modify the global content registry. `content.yaml` records mission-local official NumPy documentation and the exact sections that support the notebook.
