# Closed-Loop Runtime

The runtime now treats learning as a state transition system rather than a sequence of content pages.

```text
MISSION + LEARNER MODEL
        |
        v
PREREQUISITES / OPEN SIDE QUESTS / RETENTION DUE
        |
        v
RETRIEVE MINIMUM CONTENT
        |
        v
BUILD / EXPERIMENT / EXPLAIN
        |
        v
EVIDENCE
        |
        +--> competency/confidence update
        +--> gate attempt
        +--> misconception/repair signal
        |
        v
GATE
        |
        +--> PASS -> retention schedule -> possible autonomy ratchet -> ADVANCE
        +--> PARTIAL/FAIL -> targeted REPAIR
        +--> blocker -> bounded ZOOM_IN side quest -> micro-assess -> return
```

## Decision semantics

`learning-os step` returns exactly one primary action:

- `START`: no mission selected.
- `CONTINUE`: mission is active but has not yet produced enough evidence to judge.
- `ZOOM_IN`: a prerequisite/blocker or open side quest must be resolved.
- `REPAIR`: evidence exists but the gate is incomplete or failed.
- `RETENTION`: a spaced retrieval event is due.
- `ADVANCE`: the mission gate passed; target is the next mission.
- `COMPLETE`: reserved for a passed final capstone.

An untouched mission is **not** a failed mission. Gate status is used for routing only after evidence exists.

## Evidence-derived competency depth

The learner-model engine uses conservative provisional rules:

- L1: evidence exists.
- L2: learner can explain it.
- L3: learner built/labbed/reviewed an artifact.
- L4: build/explanation plus independent no-AI transfer.
- L5: L4 evidence demonstrated in a review/design context.

These levels are evidence summaries, not immutable truth. Human review may override them.

## Retention

A passed gate schedules mission competencies at approximately 7, 21, and 90 day retrieval intervals. Failed retention creates a short repair interval rather than silently lowering mastery.

## Autonomy ratchet

AI implementation autonomy increases only when a successful no-AI gate, successful transfer, and successful review are all evidenced. Negative independence signals reduce autonomy first. The ratchet is intentionally slower than content progression.

## Side quests

A side quest records mission, blocker/target, reason, exact return target, time budget, and micro-assessment. Default side quests cannot exceed 120 minutes. Only a PASS closes the quest; otherwise the main mission remains anchored but paused at the blocker.
