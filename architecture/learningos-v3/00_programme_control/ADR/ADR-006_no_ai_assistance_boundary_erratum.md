# ADR-006: No-AI Assistance Boundary Erratum

## Status

Accepted at R0 for G3 implementation conformance.

## Context

WP-116 and WP-125 previously implied that LearningOS could prove or technically block all external AI assistance. A local application cannot reliably observe or control other devices, services, people, or unmonitored channels. Implementing that claim would create false assurance and conflict with the controlling execution plan.

## Decision

For a recorded No-AI attempt, LearningOS must:

- disable its tutor, hints, completions, answer-revealing surfaces and other in-product assistance;
- record the assistance policy, attempt identity, timestamps, curriculum version and runner version;
- invalidate the attempt if the LearningOS lock is bypassed or fails;
- describe certification as "LearningOS assistance unavailable during the recorded attempt".

LearningOS must not claim that it proves the absence of external assistance or polices external devices and services. Telemetry may be retained as scoped diagnostic evidence only when lawful and disclosed; it is not universal proof.

## Consequences

- G3 authentication/security work enforces the product boundary only.
- Mission evidence contracts use bounded certification wording.
- Later proctoring or institutional controls require a separate explicit architecture and privacy decision.
