# WP-141: Security and Isolation Architecture

## 1. Threat Model & Boundaries
- **Threat 1: Malicious or Accidentally Destructive Student Code**:
  - Worker processes are restricted from deleting system directories or modifying the parent curriculum repository.
- **Threat 2: Unauthenticated Local API Access**:
  - The local API server generates a transient bearer token stored in `~/.learningos/.auth_token` that the React frontend includes in all HTTP/WS headers.
- **Threat 3: Plaintext Secret Exposure**:
  - Provider API keys (e.g. OpenAI/Anthropic keys) are stored in the OS Keychain (via `keyring`) or encrypted in `~/.learningos/config.json`. Never returned over client API responses.
