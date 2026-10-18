import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_router import MODEL_SPECS, ModelRouter, RouterError, fuse_results


def make_results(intervals=None):
    intervals = intervals or {}
    results = {}
    for model_id, _, model, variant in MODEL_SPECS:
        segments = [dict(start=start, end=end, score=0.9)
                    for start, end in intervals.get(model_id, [])]
        results[model_id] = dict(model=model, checkpoint_variant=variant,
                                 sample_rate=16000, duration=2.0, segments=segments,
                                 score=0.9 if segments else 0.1,
                                 label='fake' if segments else 'real', is_fake=bool(segments))
    return results


class FusionTests(unittest.TestCase):
    def test_three_models_must_agree_in_time(self):
        result = fuse_results(make_results({'cfprf-ps': [(0.4, 1.6)],
                                           'bam-wavlm': [(0.32, 1.76)],
                                           'sal-wavlm': [(0.48, 1.92)]}))
        self.assertEqual(result['segments'], [dict(start=0.48, end=1.6, score=0.6)])
        self.assertEqual(result['score'], 0.6)
        self.assertTrue(result['is_fake'])

    def test_disjoint_predictions_do_not_make_temporal_majority(self):
        result = fuse_results(make_results({'cfprf-had': [(0.0, 0.5)],
                                           'bam-wavlm': [(0.5, 1.0)],
                                           'sal-wavlm': [(1.0, 1.5)]}))
        self.assertFalse(result['is_fake'])
        self.assertEqual(result['score'], 0.2)

    def test_overlapping_proposals_only_count_one_vote(self):
        result = fuse_results(make_results({'cfprf-had': [(0.0, 1.5), (0.5, 2.0)],
                                           'bam-wavlm': [(0.0, 2.0)]}))
        self.assertFalse(result['is_fake'])
        self.assertEqual(result['score'], 0.4)

    def test_merge_contiguous_intervals_with_changing_support(self):
        result = fuse_results(make_results({'cfprf-had': [(0.5, 1.0)],
                                           'cfprf-ps': [(0.0, 2.0)],
                                           'bam-wavlm': [(0.0, 2.0)],
                                           'sal-wavlm': [(0.0, 2.0)]}))
        self.assertEqual(result['segments'], [dict(start=0.0, end=2.0, score=0.8)])

    def test_missing_model_is_not_silently_ignored(self):
        results = make_results()
        results.pop('sal-wavlm')
        with self.assertRaises(RouterError):
            fuse_results(results)

    def test_reject_invalid_or_inconsistent_outputs(self):
        mutations = [dict(duration=1.0), dict(score=float('nan')), dict(sample_rate=44100),
                     dict(checkpoint_variant='PS'), dict(label='fake'),
                     dict(segments=[dict(start=-0.1, end=0.5, score=0.8)])]
        for mutation in mutations:
            results = make_results()
            results['cfprf-had'].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(RouterError):
                fuse_results(results)


class RouterTests(unittest.TestCase):
    def test_router_calls_all_five_and_returns_one_result(self):
        results = make_results()
        calls = []
        def run(spec, audio, directory):
            calls.append(spec[0])
            self.assertTrue(audio.is_absolute())
            return copy.deepcopy(results[spec[0]])
        with tempfile.NamedTemporaryFile() as audio:
            with patch.object(ModelRouter, '_run_model', side_effect=run):
                result = ModelRouter().predict_file(audio.name)
        self.assertEqual(calls, [spec[0] for spec in MODEL_SPECS])
        self.assertEqual(len(result['model_results']), 5)

    def test_model_timeout_fails_the_request(self):
        with patch('model_router.subprocess.run', side_effect=subprocess.TimeoutExpired('model', 1)), \
             patch('pathlib.Path.is_file', return_value=True):
            with self.assertRaisesRegex(RouterError, 'timed out'):
                ModelRouter(timeout=1)._run_model(MODEL_SPECS[0], Path('audio.wav'), Path('/tmp'))

    def test_model_failure_does_not_return_partial_vote(self):
        with tempfile.NamedTemporaryFile() as audio:
            with patch.object(ModelRouter, '_run_model', side_effect=RouterError('model failed')):
                with self.assertRaises(RouterError):
                    ModelRouter().predict_file(audio.name)


if __name__ == '__main__':
    unittest.main()
