import argparse
import contextlib
import json
from pathlib import Path
import sys

from nes2net_runtime import Nes2NetDetector


def main():
    parser = argparse.ArgumentParser(description='Nes2Net-X single-file detection with original raw scores')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--test-mode', choices=['full', '4s'], default='full')
    parser.add_argument('--threshold', type=float, help='Optional calibrated raw bonafide-logit threshold; lower means fake')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--xlsr-checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = Nes2NetDetector(device=args.device, threads=args.threads, test_mode=args.test_mode,
                                    threshold=args.threshold, checkpoint=args.checkpoint,
                                    xlsr_checkpoint=args.xlsr_checkpoint)
            result = model.predict_file(args.audio)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding='utf-8')
        else:
            print(payload, end='')
    except (OSError, RuntimeError, ValueError) as error:
        print(f'Nes2Net error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
