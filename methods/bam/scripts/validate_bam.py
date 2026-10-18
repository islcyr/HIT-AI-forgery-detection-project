"""Real-weight smoke checks, including reuse and partial final frames."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[1]
sys.path.insert(0, str(METHOD))
from bam_runtime import BAMDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    start = time.monotonic()
    detector = BAMDetector(device=args.device)
    report = {'device': str(detector.device), 'strict_checkpoint_load': True,
              'load_seconds': time.monotonic() - start, 'checks': []}
    samples = ROOT / 'voxguard-fullstack/samples/audio'
    for name in ['CON_D_0023779.wav', 'LA_D_1049549.wav',
                 'dev01-xttsv2-full_1462_170138_000003_000005.wav']:
        result = detector.predict_file(samples / name, include_frames=True)
        (METHOD / 'results' / f'{Path(name).stem}.json').write_text(
            json.dumps(result, indent=2, allow_nan=False) + '\n')
        check_result(result)
        report['checks'].append({'audio': name, 'label': result['label'],
                                 'score': result['score'], 'segments': result['segments']})
    for length in [400, 2560, 2561, 340000]:
        result = detector.predict(np.zeros(length, dtype=np.float32), 16000, include_frames=True)
        check_result(result)
        report['checks'].append({'synthetic_silence_samples': length, 'frames': len(result['frame_fake_scores'])})
    report['elapsed_seconds'] = time.monotonic() - start
    if detector.device.type == 'cuda':
        import torch
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(detector.device)
    path = METHOD / 'results' / f'validation_{detector.device.type}.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(path)


def check_result(result):
    import math
    assert len(result['frame_fake_scores']) == math.ceil(result['duration'] / 0.16)
    assert len(result['frame_boundary_scores']) == len(result['frame_fake_scores'])
    assert 0 <= result['score'] <= 1
    assert result['is_fake'] == (result['score'] > 0.5)
    for segment in result['segments']:
        assert 0 <= segment['start'] < segment['end'] <= result['duration']
    assert all(0 <= score <= 1 for score in result['frame_fake_scores'])


if __name__ == '__main__':
    main()
