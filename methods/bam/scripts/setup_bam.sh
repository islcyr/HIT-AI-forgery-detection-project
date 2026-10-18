#!/usr/bin/env bash
set -euo pipefail
BAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$BAM_ROOT/../.." && pwd)"
BAM_BASE_PYTHON="${BAM_BASE_PYTHON:-python3.10}"
BAM_ENV="$PROJECT_ROOT/.envs/bam"
if [[ ! -x "$BAM_ENV/bin/python" ]]; then
    "$BAM_BASE_PYTHON" -m venv --system-site-packages "$BAM_ENV"
fi
# This server's py310 base supplies torch/torchaudio 2.5.1+cu121.
# Install explicitly only if that compatible pair is not already available.
if ! "$BAM_ENV/bin/python" -c 'import torch, torchaudio; assert torch.__version__ == "2.5.1+cu121"; assert torchaudio.__version__ == "2.5.1+cu121"'; then
    "$BAM_ENV/bin/python" -m pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
fi
# Import only s3prl's WavLM implementation, avoiding hub-wide optional packages.
"$BAM_ENV/bin/python" -m pip install --no-deps -r "$BAM_ROOT/requirements-bam.txt"
"$BAM_ENV/bin/python" "$BAM_ROOT/scripts/prepare_weights.py"
"$BAM_ENV/bin/python" -c 'from s3prl.upstream.wavlm.expert import UpstreamExpert'
cd "$BAM_ROOT"
"$BAM_ENV/bin/python" -m unittest discover -s tests -v
