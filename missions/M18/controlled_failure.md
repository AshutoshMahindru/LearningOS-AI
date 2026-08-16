# M18 Controlled Failure — Manufacture a “Significant” Win

## Seeded failure

The notebook generates 20 independent comparisons when every null hypothesis is
true, then filters the table to `p < 0.05`. With the committed seed, apparently
exciting results survive. A deliberately bad analyst reports only those rows,
changes the story to match their signs, and calls the result “statistically proven.”
This is a cherry-pick: the selected rows are presented without their comparison
family.

This is a controlled simulation. Never use the failure procedure to select claims
from real stakeholder data.

## Required diagnosis

Document:

1. **Symptom:** a small p-value appears even though the simulation contains no
   real effects.
2. **Mechanism:** repeated opportunities plus selective reporting inflate the
   chance of at least one false positive; the nominal 0.05 threshold applies to a
   single planned test, not an undisclosed search.
3. **Trace:** show the full comparison count, full result family, minimum p-value,
   selection line, and simulated familywise false-positive rate.
4. **Why it fooled us:** a plausible post-hoc story and one clean result conceal
   the discarded comparisons and researcher degrees of freedom.
5. **Why a smaller p-value is insufficient:** optional stopping, changed metrics,
   exclusions, and repeated segment searches create more unreported opportunities.

## Repair and verification

Before rerunning:

- declare one primary metric, estimand, direction, sample-size/stopping rule, and
  comparison family;
- retain every measured outcome and analysis attempt in the audit trail;
- use a family-appropriate control such as Bonferroni for strict familywise error
  or Benjamini–Hochberg when false-discovery-rate control matches the decision;
- separate exploration from confirmation and collect fresh data for a newly formed
  hypothesis; and
- report point estimates, uncertainty intervals, practical effect sizes,
  assumptions, and all planned comparisons.

Verify the repair by calculating the pre-declared adjusted threshold, showing that
the seeded nominal wins do not pass it, and explaining why correction reduces but
does not cure bias, confounding, poor measurement, or causal ambiguity.
