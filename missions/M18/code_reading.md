# M18 Code Reading

Read `simulations/M18/statistical_inference.py` before running the relevant
notebook cells. Do not begin by paraphrasing function names.

## Trace `bootstrap_differences`

1. Mark the resampling unit and explain why A and B are resampled separately.
2. Trace one resample by hand with the tiny vectors `A=[0, 0, 1]` and
   `B=[0, 1, 1]`.
3. Identify the state controlled by the seed and the statistic appended each time.
4. Explain why resampling these observations cannot correct selection bias,
   confounding, measurement error, or dependence between sessions.
5. Predict what changes if the number of resamples grows while sample sizes do not.

## Trace `permutation_test`

1. State the null-world operation represented by pooling and shuffling labels.
2. Explain why the group sizes remain fixed.
3. Explain why the comparison uses the absolute difference.
4. Explain the `+1` correction in the numerator and denominator.
5. Name the exchangeability assumption and one realistic way it could fail.

## Trace `cherry_pick_null_comparisons`

1. Find where the comparison family is created and where results are filtered.
2. Identify exactly which information a misleading report would omit.
3. Calculate the Bonferroni threshold for 20 tests before running the code.
4. Propose an interface change that forces callers to declare the metric family and
   correction before any p-values are visible.

Evidence must include a manual trace, one prediction that was wrong, and the
revised explanation—not merely the functions' returned values.
