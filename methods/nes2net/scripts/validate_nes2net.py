"""Validate actual checkpoint inference; no threshold calibration or accuracy claim."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[1]
sys.path.insert(0, str(METHOD))
from nes2net_runtime import Nes2NetDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    start = time.monotonic()
    model = Nes2NetDetector(device=args.device)
    report = {'device': str(model.device), 'strict_state_entries': model.state_entry_count,
              'load_seconds': time.monotonic() - start, 'checks': []}
    output = METHOD / 'results'
    output.mkdir(exist_ok=True)
    samples = ROOT / 'voxguard-fullstack/samples/audio'
    for name in ['CON_D_0000029.wav', 'LA_D_1049549.wav',
                 'dev01-xttsv2-full_1462_170138_000003_000005.wav']:
        result = model.predict_file(samples / name)
        repeat = model.predict_file(samples / name)
        assert result == repeat, 'Repeated eval-mode inference changed.'
        assert np.isfinite(result['score']) and result['is_fake'] is None
        assert result['segments'] is None and not result['supports_localization']
        assert result['evaluated_end_seconds'] == result['duration']
        (output / f'{Path(name).stem}_nes2net_{model.device.type}.json').write_text(
            json.dumps(result, indent=2, allow_nan=False) + '\n')
        report['checks'].append({'audio': name, 'raw_bonafide_score': result['score'], 'repeat_equal': True})
    for length in [400, 64000, 64001, 340000]:
        result = model.predict(np.zeros(length, dtype=np.float32), 16000)
        assert result['processed_samples'] == length
        report['checks'].append({'full_mode_silence_samples': length, 'score_finite': np.isfinite(result['score']).item()})
    model.test_mode = '4s'
    for length in [400, 64001]:
        result = model.predict(np.zeros(length, dtype=np.float32), 16000)
        assert result['processed_samples'] == 64000
        assert result['evaluated_end_seconds'] == min(length / 16000, 4)
        report['checks'].append({'four_seconds_mode_samples': length, 'processed_samples': 64000})
    report['elapsed_seconds'] = time.monotonic() - start
    path = output / f'validation_{model.device.type}.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(path)


if __name__ == '__main__':
    main()
