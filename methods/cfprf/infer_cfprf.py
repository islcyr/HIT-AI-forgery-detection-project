"""From this method directory: bash scripts/cfprf.sh audio.wav --output result.json"""
import argparse
import json
from pathlib import Path
import sys

from audio_forgery import CFPRFDetector


def main():
    parser = argparse.ArgumentParser(description='CFPRF single-file localization')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--variant', choices=['HAD', 'PS', 'LAVDF'], default='HAD')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda:0, cuda:1')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--include-frames', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        detector = CFPRFDetector(variant=args.variant, device=args.device, threads=args.threads)
        result = detector.predict_file(args.audio, include_frames=args.include_frames)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f'CFPRF error: {error}', file=sys.stderr)
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
