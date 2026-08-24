# M42 Controlled Failure Modes

1. **Context Window Overflow**: Truncates oldest conversation history while preserving core system instructions.
2. **Tool Execution Timeout**: Aborts hanging tool call after 500ms SLA and provides degraded response.
3. **Evaluation Gate Failure**: Rejects deployment if accuracy score drops below 0.80.
