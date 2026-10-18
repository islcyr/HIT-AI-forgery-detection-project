"""One persistent model in its original environment; JSON-lines IPC on stdio."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    model_id, device = sys.argv[1:3]
    method = model_id.split('-')[0]
    sys.path.insert(0, str(ROOT / 'methods' / method))
    with contextlib.redirect_stdout(sys.stderr):
        if method == 'cfprf':
            from audio_forgery import CFPRFDetector
            model = CFPRFDetector(variant=model_id.split('-')[1].upper(), device=device)
        elif method == 'bam':
            from bam_runtime import BAMDetector
            model = BAMDetector(device=device)
        else:
            from sal_runtime import SALDetector
            model = SALDetector(variant='WavLM', device=device)
    import soundfile as sf
    print(json.dumps({'ready': True, 'model_id': model_id, 'pid': os.getpid(), 'model_load_count': 1}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            wave, rate = sf.read(io.BytesIO(base64.b64decode(request['audio'], validate=True)), dtype='float32')
            with contextlib.redirect_stdout(sys.stderr):
                result = model.predict(wave, rate)
            response = {'result': result}
        except Exception as error:
            import traceback
            traceback.print_exc(file=sys.stderr)
            response = {'error': str(error)}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
