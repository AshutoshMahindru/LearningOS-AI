# M22 Implementation Review Brief

## Review decision requested

Approve M22 as the V05 layer-opening package: an offline, deterministic
neuron and dense-layer mission that inherits M21's black-box evidence and
explicitly defers multi-layer inference, gradients, and autograd.

This is an implementation review, not learner sign-off.

## System under review

- hand-computable two-input neuron fixture
- ReLU / sigmoid / tanh on a shared pre-activation sequence
- row-batch dense layer `Y = activation(X @ W + b)`
- two-layer affine composition with and without a hidden nonlinearity
- named orientation and activation-boundary defects
- NumPy-only runtime; no secrets; no network

## Required reviewer challenges

- verify M22 is blocked by M21 and hands off to M23 without opening backprop;
- reproduce the reference neuron (`z = 0.5`) by independent arithmetic;
- confirm bias ablation and activation sweep keep the declared invariants;
- confirm singleton and batch rows agree;
- confirm identity-hidden composition equals `collapsed_affine`;
- reproduce the controlled defect and its smallest repair;
- search the notebook for `backward`, `grad`, `softmax`, `torch`, and `coefs_`;
- verify source notebook IDs are unique, outputs empty, and labs-cwd import works;
- confirm learner evidence, ADR decisions, and completion remain unpopulated.

## Acceptance criteria

- fresh-kernel Restart + Run All succeeds from repo root and from `labs/`;
- mission unittest and pytest pass in the M22 environment;
- bare repository unittest discovery stays green (`skipUnless` on NumPy);
- no shared registry, tracking, root README, workflow, or lab-status file is changed.
