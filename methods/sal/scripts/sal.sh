#!/usr/bin/env bash
set -euo pipefail
SAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SAL_ROOT/../.." && pwd)"
exec "$PROJECT_ROOT/.envs/sal/bin/python" "$SAL_ROOT/infer_sal.py" "$@"
