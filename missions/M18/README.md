# M18 — Decide What Data Can Tell You

Variant B converted 14% of 600 fictional checkout sessions; variant A converted
12% of 600. Is the observed +2 percentage-point difference a repeatable signal,
ordinary sampling noise, or still unresolved? This mission builds a defensible
answer without hiding uncertainty behind a single p-value.

## Learner flow

1. Record a prediction before every simulation or comparison.
2. Describe the observed samples with mean, median, variance, and distributions.
3. Repeatedly sample a known population to see sampling variation and standard
   error rather than merely memorizing formulas.
4. estimate B − A, then pair it with analytic and bootstrap confidence intervals.
5. Report absolute and relative effect sizes before interpreting a hypothesis test.
6. Use a permutation test only after stating its null, assumptions, and decision use.
7. Reject a causal story supported only by correlation.
8. Trigger the controlled failure by searching many null comparisons, diagnose the
   cherry-pick, and repair the analysis with a pre-specified comparison family.
9. Complete the no-AI transfer gate and the ADR prompt.

## Run locally

From the repository root, install `requirements/m18.txt`, open
`labs/M18_statistical_inference.ipynb`, and use **Restart Kernel and Run All**. The
lab is CPU-only, uses committed synthetic fixtures, needs no secret or paid API,
and makes no runtime network request.

The notebook is an instructional instrument, not a learner-completion record.
Predictions, interpretations, and decisions belong in separate learner evidence;
none are pre-populated in this repository.
