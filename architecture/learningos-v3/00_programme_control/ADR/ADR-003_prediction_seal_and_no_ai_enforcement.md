# ADR-003: Prediction Seal & Runtime No-AI Enforcement

## Status
ACCEPTED (Controlling baseline for LearningOS V3)

## Context
In LearningOS V2, experiment results could potentially be observed before committing a hypothesis, and No-AI mode was handled superficially via UI-level toggles rather than runtime locks.

## Decision
1. **Cryptographic Prediction Seal**: In experiment stages (Predict &rarr; Commit &rarr; Run &rarr; Observe &rarr; Explain), the execution backend strictly rejects any execution request until a signed, timestamped prediction object has been recorded in the database.
2. **Runtime No-AI Lock**: During No-AI assessment and transfer stages, the API gateway rejects tutor prompts, disables hint endpoints, and verifies that the submitted solution contains zero AI telemetry tokens. The resulting evidence is flagged with `assistance_level = "NO_AI_CERTIFIED"`.

## Consequences
- Positive: Guaranteed academic rigor and authentic competency measurement.
- Positive: Evidence provenance can be verified and audited independently.
