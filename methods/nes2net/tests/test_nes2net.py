from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nes2net_runtime.detector import make_result, prepare_audio, validate_audio


class Nes2NetTests(unittest.TestCase):
    def test_input_contract(self):
        for wave, rate in [(np.zeros(400), 44100), (np.zeros((400, 2)), 16000),
                           (np.zeros(400, dtype=np.int16), 16000), (np.full(400, np.nan), 16000),
                           (np.zeros(399), 16000), (np.zeros(340001), 16000)]:
            with self.subTest(shape=wave.shape, rate=rate), self.assertRaises(ValueError):
                validate_audio(wave, rate)

    def test_full_preserves_audio_and_four_seconds_matches_demo(self):
        wave = np.linspace(-0.2, 0.2, 70000, dtype=np.float32)
        np.testing.assert_array_equal(prepare_audio(wave, 'full'), wave)
        np.testing.assert_array_equal(prepare_audio(wave, '4s'), wave[:64000])
        short = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(prepare_audio(short, '4s')[:9], np.tile(short, 3))
        self.assertEqual(len(prepare_audio(short, '4s')), 64000)

    def test_no_implicit_half_probability_threshold(self):
        result = make_result(torch.tensor([[9., 8.]]), 80000, 80000, 'full', None)
        self.assertEqual(result['score'], 8.0)
        self.assertEqual(result['label'], 'unclassified')
        self.assertIsNone(result['is_fake'])
        self.assertIsNone(result['segments'])
        self.assertFalse(result['supports_localization'])

    def test_raw_threshold_direction_and_equality(self):
        logits = torch.tensor([[9., -0.3]])
        self.assertTrue(make_result(logits, 80000, 64000, '4s', 0)['is_fake'])
        self.assertFalse(make_result(logits, 80000, 64000, '4s', -1)['is_fake'])
        self.assertFalse(make_result(torch.tensor([[2., 0.]]), 400, 400, 'full', 0)['is_fake'])
        self.assertEqual(make_result(logits, 80000, 64000, '4s', None)['evaluated_end_seconds'], 4.0)

    def test_invalid_logits(self):
        for logits in [torch.tensor([[float('nan'), 0.]]), torch.zeros(1, 3)]:
            with self.assertRaises(RuntimeError):
                make_result(logits, 400, 400, 'full', None)


if __name__ == '__main__':
    unittest.main()
