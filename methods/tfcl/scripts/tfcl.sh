#!/usr/bin/env bash
set -euo pipefail
TFCL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$TFCL_ROOT/../.." && pwd)"
exec "$PROJECT_ROOT/.envs/tfcl/bin/python" "$TFCL_ROOT/infer_tfcl.py" "$@"
