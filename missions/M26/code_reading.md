# Code reading — failure injection beside the clean loop

Read `prepare_fault`, `apply_model_hooks`, `claimed_eval`,
`repair_prepared`, and `public_symptoms` in
`missions/M26/dl_failure_lab.py` next to M25's `train_model`. M26's
code-reading target is the **diagnosis contract**:

1. M25 still owns `zero_grad → forward → loss → backward → step`
2. M26 clones the fixture, then changes one object (labels, a feature
   column, learning rate, `requires_grad`, width, Dropout, or the
   evaluation call)
3. `run_prepared` passes the prepared model, optimizer, and splits into
   `m25.train_model`; it does not rewrite the loop
4. a hidden run's public symptoms omit defect, category, and knobs
5. data faults are visible in the tensors; optimization faults are
   visible in the loss trace; gradient-flow faults are visible per
   parameter; capacity/regularization faults are visible as underfit;
   evaluation faults are visible on a frozen checkpoint
6. `repair_prepared` edits the **same** `PreparedFault` objects
7. evaluation repairs must not call `optimizer.step` or retrain

Before running the code-reading cell, predict:

- which object `label_shuffle` mutates, and which splits stay honest
- how a frozen `fc1` can still produce a falling train loss
- why `claimed_eval` can look better than held-out without training
- what `public_symptoms` must omit on a hidden run
- why repairing evaluation on the same checkpoint is smaller than
  changing width, optimizer, or data

Do **not** open language-model internals. If a failure is in data,
optimization, gradient flow, capacity, or evaluation on this loop,
stay there.
