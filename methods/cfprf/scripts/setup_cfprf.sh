#!/usr/bin/env bash
set -euo pipefail
CFPRF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$CFPRF_ROOT/../.." && pwd)"
cd "$PROJECT_ROOT"
CFPRF_BASE_PYTHON="${CFPRF_BASE_PYTHON:-python3.10}"
CFPRF_ENV="$PROJECT_ROOT/.envs/cfprf"
if [[ ! -x "$CFPRF_ENV/bin/python" ]]; then
    "$CFPRF_BASE_PYTHON" -m venv --system-site-packages "$CFPRF_ENV"
fi
"$CFPRF_ENV/bin/python" -m pip install 'pip==23.3.2'
"$CFPRF_ENV/bin/python" -m pip install 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'torchaudio==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
"$CFPRF_ENV/bin/python" -m pip install -r "$CFPRF_ROOT/requirements-cfprf.txt"
if [[ ! -d third_party/fairseq/.git ]]; then
    git clone https://github.com/pytorch/fairseq.git third_party/fairseq
    git -C third_party/fairseq checkout a54021305d6b3c4c5959ac9395135f63202db8f1
fi
[[ "$(git -C third_party/fairseq rev-parse HEAD)" == a54021305d6b3c4c5959ac9395135f63202db8f1 ]]
# Wav2vec2 inference does not use fairseq's native training extensions.
READTHEDOCS=1 "$CFPRF_ENV/bin/python" -m pip install --no-build-isolation --no-deps -e third_party/fairseq
cd "$CFPRF_ROOT"
"$CFPRF_ENV/bin/python" -m unittest discover -s tests -v
