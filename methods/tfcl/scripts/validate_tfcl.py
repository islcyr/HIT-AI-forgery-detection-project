"""Real-weight TFCL deployment validation, not a benchmark reproduction."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[1]
sys.path.insert(0, str(METHOD))
from tfcl_runtime import TFCLDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    start = time.monotonic()
    model = TFCLDetector(device=args.device)
    report = {'device': str(model.device), 'strict_state_entries': model.state_entry_count,
              'load_seconds': time.monotonic() - start, 'checks': []}
    output = METHOD / 'results'
    output.mkdir(exist_ok=True)
    samples = ROOT / 'voxguard-fullstack/samples/audio'
    for name in ['CON_D_0000029.wav', 'LA_D_1049549.wav',
                 'dev01-xttsv2-full_1462_170138_000003_000005.wav']:
        result = model.predict_file(samples / name)
        repeat = model.predict_file(samples / name)
        assert result == repeat, 'Repeated inference changed at fixed seed.'
        assert 0 <= result['score'] <= 1
        assert not result['supports_localization'] and result['segments'] is None
        assert result['windows'][-1]['end'] == result['duration']
        (output / f'{Path(name).stem}_tfcl_{model.device.type}.json').write_text(
            json.dumps(result, indent=2, allow_nan=False) + '\n')
        report['checks'].append({'audio': name, 'label': result['label'], 'score': result['score'],
                                 'repeat_equal': True, 'windows': len(result['windows'])})
    for length in [400, 64600, 64601, 340000]:
        result = model.predict(np.zeros(length, dtype=np.float32), 16000)
        assert len(result['windows']) == (length + 64599) // 64600
        assert result['windows'][-1]['end'] == result['duration']
        report['checks'].append({'synthetic_silence_samples': length, 'windows': len(result['windows'])})
    report['elapsed_seconds'] = time.monotonic() - start
    path = output / f'validation_{model.device.type}.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(path)


if __name__ == '__main__':
    main()
