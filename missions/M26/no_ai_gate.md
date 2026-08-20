# No-AI gate — diagnose a fresh hidden fault

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Use a **fresh seed** that is not the notebook's practice seed or Chaos
Day seed. Call the local harness (or an equivalent trace) and work from
symptoms. Do **not** copy the committed notebook cells verbatim.

## Part A: ranked hypotheses before edits

Write a diagnosis record with empty repair fields first:

- symptom (what you can see without naming a cause)
- at least four hypotheses spanning data, optimization, gradient flow,
  architecture/regularization, and evaluation
- the cheapest next experiment and how each outcome would reorder the
  list

Leave the root-cause line blank until the experiment has run.

## Part B: discriminating experiment

Run one cheap experiment (tiny-subset overfit, fixture-block label
agreement, per-column scales, per-parameter grads, or honest vs claimed
validation). Update the ranking. Descend into activations only if that
experiment justifies it.

## Part C: smallest repair and regression

Apply the smallest repair to the **broken objects** (restore labels on
the same tensor, restore the learning rate on the same config, unfreeze
the same parameters, restore width, or restore the evaluation call on
the same checkpoint). Rerun the original evidence. State one tempting
large change (new architecture, new optimizer family, shuffle all
splits) and why it would be unjustified.

Pass requires independent diagnosis and an oral defense. Leave all learner responses unfilled in the repository.
