#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LEARNINGOS_PYTHON:-python3}"

exec "$PYTHON_BIN" "$REPO_ROOT/tools/platform/dev.py" "$@"
