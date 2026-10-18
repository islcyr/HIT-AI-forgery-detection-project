#!/usr/bin/env bash
set -euo pipefail
CFPRF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$CFPRF_ROOT/../.." && pwd)"
exec "$PROJECT_ROOT/.envs/cfprf/bin/python" "$CFPRF_ROOT/infer_cfprf.py" "$@"
