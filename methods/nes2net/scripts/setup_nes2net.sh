#!/usr/bin/env bash
set -euo pipefail
METHOD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$METHOD_ROOT/../.." && pwd)"
NES2NET_BASE_PYTHON="${NES2NET_BASE_PYTHON:-python3.10}"
METHOD_ENV="$PROJECT_ROOT/.envs/nes2net"
if [[ ! -x "$METHOD_ENV/bin/python" ]]; then
    "$NES2NET_BASE_PYTHON" -m venv --system-site-packages "$METHOD_ENV"
fi
"$METHOD_ENV/bin/python" -m pip install pip==23.3.2
if ! "$METHOD_ENV/bin/python" -c 'import torch, torchaudio; assert torch.__version__ == "2.5.1+cu121"; assert torchaudio.__version__ == "2.5.1+cu121"'; then
    "$METHOD_ENV/bin/python" -m pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
fi
"$METHOD_ENV/bin/python" -m pip install -r "$METHOD_ROOT/requirements-nes2net.txt"
[[ "$(git -C "$PROJECT_ROOT/third_party/fairseq" rev-parse HEAD)" == a54021305d6b3c4c5959ac9395135f63202db8f1 ]]
READTHEDOCS=1 "$METHOD_ENV/bin/python" -m pip install --no-build-isolation --no-deps -e "$PROJECT_ROOT/third_party/fairseq"
cd "$METHOD_ROOT"
"$METHOD_ENV/bin/python" -m unittest discover -s tests -v
