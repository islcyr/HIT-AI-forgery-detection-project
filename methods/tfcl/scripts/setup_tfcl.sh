#!/usr/bin/env bash
set -euo pipefail
TFCL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$TFCL_ROOT/../.." && pwd)"
TFCL_BASE_PYTHON="${TFCL_BASE_PYTHON:-python3.10}"
TFCL_ENV="$PROJECT_ROOT/.envs/tfcl"
if [[ ! -x "$TFCL_ENV/bin/python" ]]; then
    "$TFCL_BASE_PYTHON" -m venv --system-site-packages "$TFCL_ENV"
fi
"$TFCL_ENV/bin/python" -m pip install pip==23.3.2
if ! "$TFCL_ENV/bin/python" -c 'import torch, torchaudio; assert torch.__version__ == "2.5.1+cu121"; assert torchaudio.__version__ == "2.5.1+cu121"'; then
    "$TFCL_ENV/bin/python" -m pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
fi
"$TFCL_ENV/bin/python" -m pip install -r "$TFCL_ROOT/requirements-tfcl.txt"
[[ "$(git -C "$PROJECT_ROOT/third_party/fairseq" rev-parse HEAD)" == a54021305d6b3c4c5959ac9395135f63202db8f1 ]]
READTHEDOCS=1 "$TFCL_ENV/bin/python" -m pip install --no-build-isolation --no-deps -e "$PROJECT_ROOT/third_party/fairseq"
cd "$TFCL_ROOT"
"$TFCL_ENV/bin/python" -m unittest discover -s tests -v
