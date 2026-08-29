# ADR-005: Isolated Process Execution Worker Boundary

## Status
ACCEPTED (Controlling baseline for LearningOS V3)

## Context
Running learner code and heavy ML computations inside the web server process risks blocking the event loop, leaking memory, or crashing the backend application during faulty learner submissions.

## Decision
1. **Subprocess Worker Architecture**: User-submitted code, notebooks, and lab runners execute inside a dedicated, isolated Python worker subprocess communicating with the API server over a typed JSON-RPC / IPC socket protocol.
2. **Resource Boundaries & Path Isolation**: Execution workers enforce explicit CPU timeouts, memory caps, and path sandboxing preventing arbitrary filesystem writes outside the designated temporary sandbox or `~/.learningos/sandbox/`.
3. **Graceful Cancellation**: Long-running or infinite loops can be cleanly terminated via `SIGTERM` / `SIGKILL` without affecting the web UI or database integrity.

## Consequences
- Positive: High system stability, resilience against student infinite loops, and clear security sandbox.
- Negative: Serialization overhead for IPC payloads (mitigated by shared memory or streaming local files for large tensors).
