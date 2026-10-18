"""Run all five deployed checkpoints and fuse their temporal decisions.

This module uses only the Python standard library. Each model runs in its own
environment, sequentially, so its dependencies and GPU allocations stay isolated.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = (
    ('cfprf-had', 'cfprf', 'CFPRF', 'HAD'),
    ('cfprf-ps', 'cfprf', 'CFPRF', 'PS'),
    ('cfprf-lavdf', 'cfprf', 'CFPRF', 'LAVDF'),
    ('bam-wavlm', 'bam', 'BAM', 'WavLM-PartialSpoof'),
    ('sal-wavlm', 'sal', 'SAL', 'WavLM'),
)
MODEL_IDS = tuple(spec[0] for spec in MODEL_SPECS)
VOTES_REQUIRED = 3


class RouterError(RuntimeError):
    pass


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fuse_results(results):
    """Vote over exact interval boundaries; no score averaging or frame upsampling."""
    if set(results) != set(MODEL_IDS):
        raise RouterError('Fusion requires successful results from exactly all five checkpoints.')
    duration = None
    for model_id, _, model_name, variant in MODEL_SPECS:
        result = results[model_id]
        if result.get('model') != model_name or result.get('checkpoint_variant') != variant:
            raise RouterError(f'{model_id}: unexpected model/checkpoint identity.')
        current_duration = result.get('duration')
        if not _number(current_duration) or not 0.025 <= current_duration <= 21.25:
            raise RouterError(f'{model_id}: invalid audio duration.')
        if duration is None:
            duration = current_duration
        if abs(current_duration - duration) > 1e-9 or result.get('sample_rate') != 16000:
            raise RouterError(f'{model_id}: inconsistent duration or sample rate.')
        if not _number(result.get('score')) or not 0 <= result['score'] <= 1:
            raise RouterError(f'{model_id}: invalid detection score.')
        if not isinstance(result.get('segments'), list):
            raise RouterError(f'{model_id}: missing segments.')
        for segment in result['segments']:
            if not isinstance(segment, dict):
                raise RouterError(f'{model_id}: invalid segment.')
            start, end, score = (segment.get(key) for key in ('start', 'end', 'score'))
            if not all(_number(value) for value in (start, end, score)):
                raise RouterError(f'{model_id}: invalid segment values.')
            if not 0 <= start < end <= duration or not 0 <= score <= 1:
                raise RouterError(f'{model_id}: segment outside audio bounds or score range.')
        expected_label = 'fake' if result['segments'] else 'real'
        if result.get('label') != expected_label or result.get('is_fake') is not bool(result['segments']):
            raise RouterError(f'{model_id}: inconsistent label and localization.')

    boundaries = sorted({0.0, duration} | {
        segment[key] for result in results.values() for segment in result['segments']
        for key in ('start', 'end')})
    timeline = []
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        # Each model has at most one vote, even if its proposals overlap.
        supporters = [model_id for model_id in MODEL_IDS if any(
            segment['start'] <= start and segment['end'] >= end
            for segment in results[model_id]['segments'])]
        votes = len(supporters)
        fraction = votes / len(MODEL_IDS)
        timeline.append({'start': start, 'end': end, 'fake_votes': votes,
                         'score': fraction, 'supporting_models': supporters})
        if votes >= VOTES_REQUIRED:
            if segments and segments[-1]['end'] == start:
                segments[-1]['end'] = end
                segments[-1]['score'] = max(segments[-1]['score'], fraction)
            else:
                segments.append({'start': start, 'end': end, 'score': fraction})
    return {
        'model': 'ensemble-5', 'fusion': 'temporal_majority_vote',
        'sample_rate': 16000, 'duration': duration,
        'is_fake': bool(segments), 'label': 'fake' if segments else 'real',
        'score': max((item['score'] for item in timeline), default=0.0),
        'score_type': 'max_temporal_model_vote_fraction',
        'decision_rule': 'at_least_three_of_five_models_mark_the_same_time_interval',
        'models_total': len(MODEL_IDS), 'votes_required': VOTES_REQUIRED,
        'segments': segments, 'vote_timeline': timeline,
        'model_results': {model_id: results[model_id] for model_id in MODEL_IDS},
    }


class ModelRouter:
    """One audio in, one fused result out. No training or preprocessing."""
    def __init__(self, device='auto', threads=4, timeout=300):
        if threads < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('threads and per-model timeout must be positive.')
        self.device = device
        self.threads = threads
        self.timeout = timeout

    def _run_model(self, spec, audio, directory):
        model_id, method, _, variant = spec
        python = ROOT / '.envs' / method / 'bin/python'
        if not python.is_file():
            raise RouterError(f'{model_id}: missing environment: {python}')
        output = directory / f'{model_id}.json'
        command = [str(python), str(ROOT / 'methods' / method / f'infer_{method}.py'),
                   str(audio), '--device', self.device, '--threads', str(self.threads),
                   '--output', str(output)]
        if method in ('cfprf', 'sal'):
            command.extend(['--variant', variant])
        try:
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                     timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise RouterError(f'{model_id}: timed out after {self.timeout} seconds.') from error
        except OSError as error:
            raise RouterError(f'{model_id}: failed to launch: {error}') from error
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip()[-3000:]
            raise RouterError(f'{model_id}: inference failed (exit {process.returncode}).\n{detail}')
        try:
            result = json.loads(output.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise RouterError(f'{model_id}: missing or invalid output JSON.') from error
        if not isinstance(result, dict):
            raise RouterError(f'{model_id}: expected a JSON object.')
        return result

    def predict_file(self, audio_path):
        audio = Path(audio_path).expanduser().resolve(strict=True)
        if not audio.is_file():
            raise ValueError(f'Audio is not a file: {audio}')
        with tempfile.TemporaryDirectory(prefix='audio-model-router-') as directory:
            results = {}
            for spec in MODEL_SPECS:
                results[spec[0]] = self._run_model(spec, audio, Path(directory))
        return fuse_results(results)


def main():
    parser = argparse.ArgumentParser(description='Run five checkpoints and return one fused localization result')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=300, help='Timeout per model, in seconds')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = ModelRouter(device=args.device, threads=args.threads,
                             timeout=args.timeout).predict_file(args.audio)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Only replace an existing result after all five models succeed.
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent,
                                             prefix='.router-', suffix='.json', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            try:
                temporary.replace(args.output)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            print(payload, end='')
    except (RouterError, ValueError, OSError) as error:
        print(f'Model router error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
