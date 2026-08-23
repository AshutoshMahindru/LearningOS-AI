# Controlled failure — loose inference controls or the wrong lever

## Failure: two outputs, or a fine-tune, blamed on the wrong cause

Use the local 4-token score fixture, seed `3201`, and an attached M31
checkpoint (`inference_ready=True`). Predict, before running, which
recorded fields would have to match before two completions can be
blamed on a model change, and which adaptation lever is smallest when
answers quote last year's site hours.

Then run one named defect.

The defective path uses one named change:

- `uncontrolled_settings`: two sampled runs share a checkpoint and
  prompt but not temperature and seed; a naive compare labels the
  token disagreement `model_changed`, or
- `wrong_adaptation`: a team proposes parameter change because Site B
  holiday hours are stale.

The teaching decoder, vocab, and checkpoint identity stay fixed. Only
sampling controls or the chosen adaptation route change.

The defect can still emit two different token strings, or a confident
fine-tune proposal. That is the point. Diagnosis comes from:

1. `InferenceConfig` fields (temperature, seed, prompt, stop, max-tokens),
2. `compare_outputs` versus `compare_outputs_naive`,
3. adaptation signals (`freshness`, `private_knowledge`, …),
4. the hand softmax of `(log 3, 0)` at `T=1` → `(0.75, 0.25)`.

## Discriminators

Uncontrolled settings: `checkpoint_id` still matches; `prompt_ids`
still match; `temperature` and `seed` differ; naive compare says
`model_changed`; controlled compare says `uncontrolled_settings`.

Wrong adaptation: generated tokens are not the issue; `chosen_route`
is `parameters` on a freshness/private-knowledge case. The rubric's
smallest sufficient lever is not a weight update.

Do not start with "try another model" or "lower the loss." Read the
config and the signals first.

## Repair rule

The smallest repair calls `repair_run` on the **broken trace** so
either both generations reuse the defective object's reference config
or the adaptation decision is recomputed from that object's signals.
Do not start two unrelated `defect="none"` runs, do not build a search
index, and do not open M34/M37 implementations.

Submit prediction, named defect, preserved fields, first divergence or
wrong route, root cause, smallest repair, verification, and the
regression that the broken path still diverges.

A repair is rejected if it opens M33-M37 mechanisms, if it is two
unrelated happy-path runs, or if it changes several healthy variables
at once.
