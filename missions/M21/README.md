# M21 — Train a Neural Network as a Black Box

M20 established a monitored optimization policy. M21 now trains a complete neural
network **before opening its internals**. The learner treats the network as a bounded
system with declared inputs, split policy, preprocessing, architecture knobs, optimizer
configuration, seeds, learning curves, validation evidence, held-out metrics, and error
patterns.

The reference fixture uses scikit-learn's bundled handwritten-digits dataset and one
seeded `MLPClassifier` pipeline. It is CPU-only, offline, deterministic where practical,
secret-free, and paid-API-free. The notebook never inspects weight matrices, neuron
activations, a manual forward pass, gradients, or backpropagation; those mechanisms are
reserved for M22-M24.

Learners predict whole-system behavior before every run, compare against a majority
baseline, verify same-seed reproducibility, test seed sensitivity, isolate undertraining
and corrupted-label failures, compare a deliberately tiny-capacity model, and inspect
held-out error structure without opening the network.

Implementation status is not learner completion. Predictions, evidence, no-AI work,
ADR decisions, mission scores, and learner sign-off remain intentionally unfilled.
