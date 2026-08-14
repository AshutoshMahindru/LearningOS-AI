# M03 — Modify Python Before Learning Python

## Mission

Start with a working Python program and make useful changes before studying Python as a collection of language features.

The learning loop is:

1. Run the working program.
2. Explain what you think it currently does.
3. Predict the effect of one small change.
4. Make the smallest useful modification.
5. Run the program again.
6. Compare prediction with observation.
7. Trace execution when the result is surprising.
8. Repair the program and explain the evidence that identified the cause.

## What you will modify

The lab exposes Python operationally through:

- values and variables;
- strings, numbers, lists and dictionaries;
- indexing and lookup;
- `if` conditions;
- `for` loops;
- function arguments and return values;
- local intermediate values;
- assertions;
- exceptions and debugging.

Formal terminology is introduced only after the learner has encountered a need for it.

## Debugging discipline

Do not repair code by random editing.

Use:

**symptom → prediction → trace → hypothesis → smallest test → repair → verification**

The mission includes NameError, TypeError, KeyError, condition-direction and off-by-one failure modes plus a controlled failure with one narrow root cause.

## Source policy

`python-tutorial` from `data/source_registry.json` is the authoritative just-in-time reference. Consult only the section required to explain behavior you have already encountered rather than reading the tutorial front-to-back first.

## V01 connection

M03 prepares the learner for the V01 Structured Data Workbench. Later data work depends on being able to inspect unfamiliar Python, change transformations safely, trace state and repair failures without rewriting an entire program.

## Completion evidence

Completion requires evidence of predictions, modifications, debugging traces, repairs, explanations, code reading and a fresh no-AI transfer task.
