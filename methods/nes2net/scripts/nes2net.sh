#!/usr/bin/env bash
set -euo pipefail
METHOD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$METHOD_ROOT/../.." && pwd)"
exec "$PROJECT_ROOT/.envs/nes2net/bin/python" "$METHOD_ROOT/infer_nes2net.py" "$@"
