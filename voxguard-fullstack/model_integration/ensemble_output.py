"""Map temporal majority output to the unchanged frontend/backend contract."""
import math
from pathlib import Path
import sys

from voxguard.adapters import Prediction

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.model_router import fuse_results


def convert_output(raw, audio, descriptor, mode):
    if raw != fuse_results(raw['model_results']):
        raise ValueError('Ensemble output differs from its five source predictions')
    if descriptor['decision_operator'] != 'gt' or descriptor['thresholds']['detection'] != 0.5 or descriptor['thresholds']['localization'] != 0.5:
        raise ValueError('Expected three-of-five majority policy')
    rate = raw['sample_rate']
    if rate != descriptor['input']['sample_rate_hz'] or not math.isclose(raw['duration'], audio.duration_sec, abs_tol=1/rate, rel_tol=0):
        raise ValueError('Ensemble input duration/rate mismatch')
    requested = mode == 'detect_and_localize'
    duration = audio.duration_sec
    segments = [{'start_sec': min(s['start'], duration), 'end_sec': min(s['end'], duration),
                 'spoof_score': s['score']} for s in raw['segments'] if s['start'] < duration]
    timeline = []
    previous = 0.0
    for interval in raw['vote_timeline']:
        end = min(interval['end'], duration)
        if end > previous:
            timeline.append({'start_sec': previous, 'end_sec': end, 'spoof_score': interval['score']})
            previous = end
    if timeline:
        timeline[-1]['end_sec'] = duration
    return Prediction({
        'decision': {'label': 'spoof' if raw['is_fake'] else 'bonafide', 'spoof_score': raw['score'],
                     'threshold': 0.5, 'threshold_operator': 'gt', 'score_type': 'uncalibrated',
                     'source_score_type': raw['score_type'], 'score_direction': 'higher_is_spoof',
                     'decision_rule': raw['decision_rule']},
        'localization': {'status': 'available' if requested else 'not_requested',
                         'score_aggregation': 'proposal', 'segments': segments if requested else None,
                         'resolution_sec': descriptor['localization_resolution_sec'] if requested else None,
                         'threshold': 0.5 if requested else None,
                         'postprocessing_version': 'ensemble-five-temporal-majority-v1' if requested else None},
        'timeline': timeline if requested else None,
    }, rate)
