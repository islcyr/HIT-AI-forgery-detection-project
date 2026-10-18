#!/usr/bin/env bash
set -euo pipefail
BAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$BAM_ROOT/../.." && pwd)"
exec "$PROJECT_ROOT/.envs/bam/bin/python" "$BAM_ROOT/infer_bam.py" "$@"
