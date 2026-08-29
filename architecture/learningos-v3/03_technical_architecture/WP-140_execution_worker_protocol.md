# WP-140: Execution Worker Protocol & Subprocess Isolation

## 1. IPC Protocol Specification
Communication between the API server and the Execution Worker occurs via JSON-RPC 2.0 over Unix Domain Socket (`/tmp/learningos_worker.sock` on macOS/Linux or named pipe on Windows).

### Sample Request:
```json
{
  "jsonrpc": "2.0",
  "id": "req_12345",
  "method": "execute_task",
  "params": {
    "module": "missions.M25.lab",
    "entrypoint": "run_training_experiment",
    "code": "def custom_optimizer(): ...",
    "parameters": { "epochs": 5, "lr": 0.01 },
    "limits": { "timeout_sec": 30, "memory_mb": 2048 }
  }
}
```

## 2. Worker Lifecycle & Watchdog
- The worker runs inside an isolated process group.
- An OS-level watchdog monitors CPU time and resident memory.
- If execution exceeds `timeout_sec`, the worker terminates with `SIGKILL`, and the API server reports a clean `status: "TIMEOUT"` structured result.
