# No-AI gate — defend inference control from a blank page

Complete this gate from a blank page without AI-generated code,
calculations, prose, or diagrams.

Do not reuse notebook outputs. Use only the numbers below.

## Fixture (fresh)

Three-class logits (not the notebook filter vector):

```
logits = (log(6), log(3), log(1))
```

Temperature `1`. Ignore the local score table; compute by hand.

Two run records, same claimed checkpoint `v07-teaching-lm-1`:

```
record A: output tokens = (alpha, beta), other fields missing
record B: output tokens = (gamma, stop), other fields missing
claim: "the model changed"
```

Scenario (not a notebook case id): a teammate wants to fine-tune
because the model does not know this week's cafeteria menu.

## Part A: greedy and temperature

1. Which class does greedy select from the three-class logits?
2. Write the three softmax probabilities at temperature 1.
3. If temperature rises, does the largest probability rise, fall, or
   stay the same? One sentence.

## Part B: filters

Keep `top-k = 2` on the same logits. Which classes remain, and what is
the renormalized mass on the greedy class?

## Part C: uncontrolled variables

The two run records are compared as if the checkpoint changed. List the
metadata that would have to be recorded and matched before that claim
is legal.

## Part D: adaptation

For the cafeteria-menu complaint, choose among prompt, retrieval,
tools, and parameters. State one reason the first lever you reject is
wrong.

## Part E: reproduce an inference result

List the fields that must appear on an `InferenceConfig` (or equivalent
provider log) to replay a sampled completion.

Pass requires greedy/filter arithmetic, an uncontrolled-settings call,
an adaptation choice, a metadata list, and an oral defense.
Leave all learner responses unfilled in the repository.
