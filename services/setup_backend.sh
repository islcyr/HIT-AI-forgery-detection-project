#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOXGUARD_BASE_PYTHON="${VOXGUARD_BASE_PYTHON:-python}"
"$VOXGUARD_BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3,11), "Backend requires Python >= 3.11"'
if [[ ! -x "$PROJECT_ROOT/.envs/voxguard/bin/python" ]]; then
    "$VOXGUARD_BASE_PYTHON" -m venv "$PROJECT_ROOT/.envs/voxguard"
fi
"$PROJECT_ROOT/.envs/voxguard/bin/python" -m pip install -r "$PROJECT_ROOT/voxguard-fullstack/backend/requirements-lock.txt"
