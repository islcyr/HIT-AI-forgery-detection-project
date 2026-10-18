"""Run a single audio through a selected SAL checkpoint."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

from sal_runtime import SALDetector


def main():
    parser = argparse.ArgumentParser(description='SAL single-file localization')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--variant', choices=['W2V2', 'WavLM'], default='WavLM')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--encoder-checkpoint', type=Path)
    parser.add_argument('--include-frames', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = SALDetector(variant=args.variant, device=args.device, threads=args.threads,
                                threshold=args.threshold, checkpoint=args.checkpoint,
                                encoder_checkpoint=args.encoder_checkpoint)
            result = model.predict_file(args.audio, include_frames=args.include_frames)
    except (ValueError, RuntimeError, OSError) as error:
        print(f'SAL error: {error}', file=sys.stderr)
        return 1
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    else:
        print(payload, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
