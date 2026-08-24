# M41 Controlled Failure Modes

1. **Cascading Retrieval Outage**: Vector store latency exceeds 500ms SLA. System degrades to cached keyword retrieval.
2. **Trust Boundary Violation**: Model attempts unauthenticated tool execution. Enforcer rejects execution at control boundary.
3. **Observability Budget Exhaustion**: Token budget depleted; request degrades gracefully to cached answer.
