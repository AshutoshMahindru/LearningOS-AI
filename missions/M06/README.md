# M06 — See the Data Before Modeling It

## Mission objective

Use visualization to interrogate a dataset before choosing or fitting a model. The learner begins with a decision question, identifies the evidence that would bear on it, and only then selects a chart.

The guided case asks:

> Which observable signals are associated with escalation in the support-ticket sample, and what data-quality or leakage risks must be resolved before any escalation model is attempted?

The notebook examines distribution, missingness, outliers, relationships, groups, target imbalance, and possible leakage. Every section keeps three kinds of statements separate:

1. **Visible observation** — what the chart or table directly displays.
2. **Inference** — a plausible interpretation that still needs checking.
3. **What cannot yet be concluded** — causal, population, or operational claims the current evidence cannot support.

## Learning loop

For each investigation:

**question → prediction → chart choice → run → visible observation → inference → limitation → next question**

Chart names are introduced only after the analytical question is stated. A chart is evidence, not decoration and not proof of causality.

## Controlled failure

The lab draws the same mean customer tenure by channel twice. One view truncates the vertical scale and visually exaggerates modest differences; the other restores a zero baseline and the sample's full tenure range. The learner must diagnose the rhetorical effect, not merely identify a plotting parameter.

## Data

- `datasets/M06/support_tickets.csv` is the guided fixture.
- `datasets/M06/community_programs_fresh.csv` is held back for the no-AI transfer gate.
- `datasets/M06/README.md` defines provenance, timing, and field meanings.

Both datasets are deterministic, synthetic, local, and safe to run without network access or credentials.

## Completion evidence

Completion requires a question log, prediction log, chart rationale, observation/inference/limitation statements, leakage diagnosis, controlled-failure diagnosis, and a fresh no-AI transfer response. Repository implementation status is not learner completion.
