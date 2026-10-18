from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tfcl_runtime.detector import TFCLDetector, WINDOW_SAMPLES, inference_rng, prepare_window, validate_audio


class TFCLTests(unittest.TestCase):
    def test_original_repeat_padding_and_crop(self):
        wave = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        padded = prepare_window(wave)
        self.assertEqual(len(padded), 64600)
        np.testing.assert_array_equal(padded[:9], np.tile(wave, 3))
        long = np.arange(64601, dtype=np.float32)
        np.testing.assert_array_equal(prepare_window(long), long[:64600])

    def test_reject_invalid_input(self):
        for wave, rate in [(np.zeros(400), 44100), (np.zeros((400, 2)), 16000),
                           (np.zeros(400, dtype=np.int16), 16000), (np.full(400, np.nan), 16000),
                           (np.zeros(399), 16000), (np.zeros(340001), 16000)]:
            with self.subTest(rate=rate, shape=wave.shape), self.assertRaises(ValueError):
                validate_audio(wave, rate)

    def test_rng_repeatable_and_global_state_restored(self):
        torch_before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        def sample():
            with inference_rng(1234, torch.device('cpu')):
                return torch.rand(4), np.random.rand(4)
        first, second = sample(), sample()
        self.assertTrue(torch.equal(first[0], second[0]))
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        self.assertEqual(numpy_before[2:], np.random.get_state()[2:])

    def test_long_audio_tail_is_processed_and_not_localized(self):
        detector = TFCLDetector.__new__(TFCLDetector)
        detector.device = torch.device('cpu')
        detector.seed, detector.threshold = 1234, 0.5
        detector.model = Mock(side_effect=[torch.tensor([[-2., 2.]]), torch.tensor([[2., -2.]])])
        result = detector.predict(np.zeros(WINDOW_SAMPLES + 400, dtype=np.float32), 16000)
        self.assertEqual(detector.model.call_count, 2)
        self.assertTrue(result['is_fake'])
        self.assertGreater(result['score'], 0.9)
        self.assertFalse(result['supports_localization'])
        self.assertIsNone(result['segments'])
        self.assertEqual(result['windows'][-1]['end'], result['duration'])


if __name__ == '__main__':
    unittest.main()
