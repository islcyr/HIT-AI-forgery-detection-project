import argparse
import contextlib
import json
from pathlib import Path
import sys

from tfcl_runtime import TFCLDetector


def main():
    parser = argparse.ArgumentParser(description='TFCL single-file utterance detection')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--xlsr-checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = TFCLDetector(device=args.device, threads=args.threads, seed=args.seed,
                                 threshold=args.threshold, checkpoint=args.checkpoint,
                                 xlsr_checkpoint=args.xlsr_checkpoint)
            result = model.predict_file(args.audio)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding='utf-8')
        else:
            print(payload, end='')
    except (ValueError, RuntimeError, OSError) as error:
        print(f'TFCL error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
