#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_ROOT/.envs/voxguard/bin/python" "$PROJECT_ROOT/services/launch.py" "$@"
