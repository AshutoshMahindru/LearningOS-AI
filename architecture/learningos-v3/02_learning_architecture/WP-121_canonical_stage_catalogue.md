# WP-121: Canonical Learning-Stage Component Catalogue

## The 11 Controlled Stage Primitives
All 42 missions in LearningOS V3 are constructed exclusively by composing these 11 canonical, reusable stage components:

1. **`orientation`**:
   - High-level mission framing, real-world context, core invariant, target competency list, and flagship release relationship.
2. **`trace_map`**:
   - Interactive whole-system architectural diagram / trace canvas. Learner maps components, data flows, and invariants.
3. **`interrogate`**:
   - Socratic interrogation of the whole system. Probing questions regarding failure modes, data shapes, and assumptions.
4. **`experiment`**:
   - Strict 5-step cycle: **Predict** &rarr; **Commit** &rarr; **Run** &rarr; **Observe** &rarr; **Explain**. Prediction is sealed before code executes.
5. **`code_reading`**:
   - Focused dissection of real, unfamiliar production or framework code (e.g. PyTorch autograd internals, NumPy vectorization).
6. **`rebuild_debug`**:
   - Guided reconstruction or debugging of a core algorithm/mechanism with incremental test feedback.
7. **`controlled_failure`**:
   - System failure is deliberately injected (e.g. exploding gradients, distribution shift, token truncation). Learner must isolate and repair the defect.
8. **`transfer_assessment`**:
   - Fresh-case challenge without AI assistance. Verifies unassisted implementation ability and transfer of conceptual model.
9. **`competency_gate`**:
   - Automated evaluation of all submitted stage evidence against the mission's evidence contract. Issues certification or generates repair plan.
10. **`reflection_adr`**:
    - Formal authoring of an Architectural Decision Record (ADR) documenting tradeoffs, rationale, and consequences.
11. **`flagship_integration`**:
    - Merging the newly mastered capability into the long-running flagship Operations Intelligence System release.
