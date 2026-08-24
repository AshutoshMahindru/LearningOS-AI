# M40 fixtures

Offline, versioned evaluation pack for **Evaluate AI Systems Systematically**.

These files are synthetic, deterministic, and authored for M40. They
invoke the bundled M34 RAG pipeline and M39 robust agent. They are not
a production benchmark, not a paid eval-SDK log, and not an M41
architecture diagram. They require no download and no network.

- `eval_pack.json` — frozen `m40.eval.v1` cases spanning RAG (citation /
  abstention) and agent (schema / termination / memory / fallback /
  idempotency). Do not retune M34 or M39 against holdout ids.
- `contaminated_pack.json` — teaching anti-fixture `m40.eval.tuned-dev`
  that dropped hard cases and relabeled the rest as holdout.
- `expected.json` — fixture suite properties (not learner evidence).
- `rubric_labels.json` — frozen hand labels for rubric calibration.
- `transfer.json` — fresh numbers for the no-AI gate.
- `freeze_expected.py` — regenerates `expected.json` from the live
  harness. Canonical tests load the frozen file.

M41 may reuse this suite. M41 must not treat this directory as a
system-architecture deliverable.
