# M08 Formal Engineering Review Brief

## Review decision requested

Decide whether the M08 regression system is acceptable as the continuous-outcome evaluation foundation for V02. Approval covers the architecture and evaluation protocol, not real-world housing deployment.

## Architecture under review

`bundled CSV → availability allow-list → deterministic train/test split → baseline + fitted pipeline → held-out predictions → MAE/RMSE/R² → residual and CV diagnosis → ADR`

The test partition is excluded from fitting and candidate development. Cross-validation runs only on the training partition. `post_sale_assessment_k` sits outside the deployable boundary and exists solely to demonstrate leakage.

## Meaningful artifacts

- `labs/M08_regression.ipynb`: executable whole-first regression and diagnosis narrative;
- `datasets/M08/housing_regression.csv`: deterministic local fixture;
- `datasets/M08/generate_dataset.py`: reproducible data generator;
- `tests/missions/test_m08.py`: package, dataset, notebook and model-contract checks;
- `missions/M08/adr_prompt.md`: required decision-record structure.

## Reviewer evidence checklist

- [ ] The mean baseline is evaluated on exactly the same held-out rows as the safe model.
- [ ] MAE and RMSE retain target units; R² is not treated as an error magnitude.
- [ ] Residual sign and visible patterns are explained.
- [ ] Cross-validation does not consume the final test partition.
- [ ] Train/CV gaps support the underfit/overfit diagnosis.
- [ ] Permutation importance includes uncertainty and non-causal caveats.
- [ ] The leakage experiment is visibly labelled invalid and repaired by an availability allow-list.
- [ ] Training-only performance is not presented as generalization.
- [ ] The ADR connects a model/evaluation decision to observed evidence.
- [ ] Restart + Run All and all named validation commands have recorded outcomes.

## Failure analysis to challenge

The controlled failure is subtle because both a random holdout and cross-validation preserve the relationship between the post-sale assessment and the target. The root cause is a violated time-of-availability boundary, not duplicated rows. Reviewers should ask how the same control would detect labels, aggregates, proxies or timestamps created after the prediction moment.

## Consequential decision

The learner must decide whether to retain the random-forest pipeline as the V02 regression baseline and which evaluation protocol governs later changes. The ADR must compare at least the mean baseline, a capacity-controlled tree or linear model, and the random forest. It must also state why a higher-scoring leaky alternative is inadmissible.

## Known uncertainty

The synthetic fixture has no temporal, geographic or social deployment validity. A production review would additionally require representative data, temporal or grouped validation, subgroup error analysis, calibration to decision costs, monitoring and governance.

## Review response rule

Every reviewer comment must be **accepted**, **rejected** or **deferred** with written reasoning. Approval does not populate learner completion evidence automatically.
