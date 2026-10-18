"""Validate both SAL variants; record failures, and exit nonzero if either fails."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[1]
sys.path.insert(0, str(METHOD))
from sal_runtime import SALDetector


def check_result(result):
    assert len(result['frame_fake_scores']) == math.ceil(result['duration'] / 0.16)
    assert all(0 <= score <= 1 for score in result['frame_fake_scores'])
    assert result['is_fake'] == (result['score'] > 0.5)
    for segment in result['segments']:
        assert 0 <= segment['start'] < segment['end'] <= result['duration']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--variants', nargs='+', choices=['W2V2', 'WavLM'], default=['W2V2', 'WavLM'])
    args = parser.parse_args()
    output = METHOD / 'results'
    output.mkdir(exist_ok=True)
    report = {'device': args.device, 'variants': {}}
    all_passed = True
    for variant in args.variants:
        checks = []
        start = time.monotonic()
        model = None
        try:
            model = SALDetector(variant=variant, device=args.device)
            load_seconds = time.monotonic() - start
            for name in ['CON_D_0000029.wav', 'LA_D_1049549.wav',
                         'dev01-xttsv2-full_1462_170138_000003_000005.wav']:
                result = model.predict_file(ROOT / 'voxguard-fullstack/samples/audio' / name, include_frames=True)
                check_result(result)
                (output / f'{Path(name).stem}_sal_{variant.lower()}.json').write_text(
                    json.dumps(result, indent=2, allow_nan=False) + '\n')
                checks.append({'audio': name, 'label': result['label'],
                               'score': result['score'], 'segments': result['segments']})
            for length in [400, 2560, 2561, 340000]:
                result = model.predict(np.zeros(length, dtype=np.float32), 16000, include_frames=True)
                check_result(result)
                checks.append({'synthetic_silence_samples': length, 'frames': len(result['frame_fake_scores'])})
            report['variants'][variant] = {'status': 'passed', 'strict_state_entries': model.state_entry_count,
                                           'load_seconds': load_seconds,
                                           'elapsed_seconds': time.monotonic() - start, 'checks': checks}
        except (RuntimeError, ValueError, OSError) as error:
            all_passed = False
            report['variants'][variant] = {'status': 'failed', 'error': str(error), 'checks': checks}
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report['all_variants_passed'] = all_passed
    path = output / f'validation_{args.device.split(":")[0]}.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(path)
    return 0 if all_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
