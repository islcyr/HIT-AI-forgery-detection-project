#!/usr/bin/env bash
set -euo pipefail
SAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SAL_ROOT/../.." && pwd)"
SAL_BASE_PYTHON="${SAL_BASE_PYTHON:-python3.10}"
SAL_ENV="$PROJECT_ROOT/.envs/sal"
if [[ ! -x "$SAL_ENV/bin/python" ]]; then
    "$SAL_BASE_PYTHON" -m venv --system-site-packages "$SAL_ENV"
fi
if ! "$SAL_ENV/bin/python" -c 'import torch, torchaudio; assert torch.__version__ == "2.5.1+cu121"; assert torchaudio.__version__ == "2.5.1+cu121"'; then
    "$SAL_ENV/bin/python" -m pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
fi
# Narrow s3prl imports avoid Lightning, Transformers, and full-hub dependencies.
"$SAL_ENV/bin/python" -m pip install --no-deps -r "$SAL_ROOT/requirements-sal.txt"
cd "$SAL_ROOT"
"$SAL_ENV/bin/python" -m unittest discover -s tests -v
