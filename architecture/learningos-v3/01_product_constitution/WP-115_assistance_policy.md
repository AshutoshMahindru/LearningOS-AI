# WP-115: Assistance Policy Matrix

## Stage-Level Assistance Matrix
Assistance rights are governed strictly by the active learning stage, enforced by the backend API and Tutor service.

| Stage Type | Socratic Guidance | Direct Hints | Autocomplete / Code Gen | Solution Reveal |
|---|---|---|---|---|
| **Orientation / System Map** | ALLOWED (Contextual) | ALLOWED | DISABLED | ALLOWED |
| **Interrogate / Decompose** | ALLOWED (Socratic Only) | RESTRICTED (After 2 attempts) | DISABLED | DISABLED |
| **Predict & Commit** | DISABLED (Unassisted) | DISABLED | DISABLED | DISABLED (Sealed) |
| **Run & Observe** | ALLOWED (Output analysis) | RESTRICTED | DISABLED | DISABLED |
| **Rebuild & Debug** | ALLOWED (Diagnostic prompts)| RESTRICTED (Point to error line)| DISABLED | DISABLED |
| **Controlled Failure** | ALLOWED (Hypothesis checks) | RESTRICTED | DISABLED | DISABLED |
| **No-AI Transfer** | **PROHIBITED (Locked)** | **PROHIBITED** | **PROHIBITED** | **PROHIBITED** |
| **Competency Gate** | **PROHIBITED (Locked)** | **PROHIBITED** | **PROHIBITED** | **PROHIBITED** |
| **Reflection / ADR** | ALLOWED (Critique & Review) | ALLOWED | DISABLED | N/A |
| **Flagship Integration** | ALLOWED (Architectural Review)| ALLOWED | RESTRICTED | N/A |

## Assistance Enforcement Invariant
The tutor engine verifies the session's active stage token. If a stage specifies `assistance: "NO_AI_REQUIRED"`, all prompt queries return HTTP `403 Forbidden` with a policy explanation.
