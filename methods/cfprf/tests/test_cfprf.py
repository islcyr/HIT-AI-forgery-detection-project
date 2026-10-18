import unittest
from unittest.mock import patch

import numpy as np
import torch

from audio_forgery.cfprf import CFPRFDetector, validate_audio
from evaluate_CFPRF import Inference_CFPRF


class FixedFDN(torch.nn.Module):
    def forward(self, wave):
        # Two forged proposals: one eligible for regression, one short.
        logits = torch.tensor([[[3., 0.], [3., 0.], [0., 3.], [3., 0.]]])
        features = torch.zeros(1, 4, 128)
        return logits, logits, features, features


class FixedPRN(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.probabilities = probabilities

    def forward(self, features, proposals, rso):
        confidence = torch.tensor(self.probabilities).reshape(-1, 1)
        regression = torch.tensor([[25., 0.], [100., 100.]])
        scored = np.copy(proposals[0])
        scored[:, 0] = confidence.numpy().ravel()
        return confidence, regression, None, [scored]


class AdapterTests(unittest.TestCase):
    def detector(self, probabilities):
        detector = CFPRFDetector.__new__(CFPRFDetector)
        detector.device = torch.device('cpu')
        detector.variant = 'HAD'
        detector.fdn = FixedFDN()
        detector.prn = FixedPRN(probabilities)
        return detector

    def test_backend_contract(self):
        self.assertEqual(validate_audio(np.zeros(16000, dtype=np.float64), 16000).dtype, np.float32)
        for wave, rate in [
            (np.zeros(16000, dtype=np.float32), 48000),
            (np.zeros((16000, 2), dtype=np.float32), 16000),
            (np.zeros(16000, dtype=np.int16), 16000),
            (np.zeros(399, dtype=np.float32), 16000),
            (np.zeros(340001, dtype=np.float32), 16000),
            (np.full(1000, np.nan, dtype=np.float32), 16000),
        ]:
            with self.subTest(shape=wave.shape, rate=rate):
                with self.assertRaises(ValueError):
                    validate_audio(wave, rate)

    def test_decoding_matches_official_evaluator(self):
        wave = np.zeros(3200, dtype=np.float32)
        for probabilities in ([0.9, 0.8], [0.2, 0.9], [0.9, 0.2], [0.5, 0.5]):
            with self.subTest(probabilities=probabilities):
                detector = self.detector(probabilities)
                result = detector.predict(wave, 16000, include_frames=True)
                batch = [(torch.from_numpy(wave)[None], ['sample'], torch.zeros(1, 4))]
                with patch('builtins.print'):
                    *_, reference = Inference_CFPRF(batch, detector.fdn, detector.prn, 20, torch.device('cpu'))
                actual = np.asarray([[s['score'], s['start'], s['end']] for s in result['segments']]).reshape(-1, 3)
                expected = np.asarray(reference['sample']).reshape(-1, 3)
                if len(expected):
                    expected = expected[np.argsort(expected[:, 1])]
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                self.assertEqual(result['is_fake'], bool(len(expected)))
                self.assertGreater(result['frame_fake_scores'][0], 0.9)
                self.assertLess(result['frame_fake_scores'][2], 0.1)

    def test_no_coarse_proposals_skips_prn(self):
        detector = self.detector([0.9, 0.9])
        def real_only(wave):
            logits = torch.tensor([[[0., 3.], [0., 3.]]])
            return logits, logits, None, None
        detector.fdn.forward = real_only
        with patch.object(detector.prn, 'forward', side_effect=AssertionError('PRN should not run')):
            result = detector.predict(np.zeros(3200, dtype=np.float32), 16000)
        self.assertEqual(result['segments'], [])
        self.assertEqual(result['score'], 0.0)
        self.assertFalse(result['is_fake'])


if __name__ == '__main__':
    unittest.main()
