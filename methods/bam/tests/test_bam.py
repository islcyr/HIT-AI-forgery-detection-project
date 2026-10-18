import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bam_runtime.detector import decode_scores, validate_audio


class InputContractTests(unittest.TestCase):
    def test_reject_backend_preprocessing_mismatches(self):
        for wave, rate in [(np.zeros(1000, dtype=np.float32), 44100),
                           (np.zeros((1000, 2), dtype=np.float32), 16000),
                           (np.zeros(1000, dtype=np.int16), 16000),
                           (np.full(1000, np.nan), 16000),
                           (np.zeros(399), 16000), (np.zeros(340001), 16000)]:
            with self.subTest(rate=rate, shape=wave.shape), self.assertRaises(ValueError):
                validate_audio(wave, rate)

    def test_no_amplitude_normalization(self):
        wave = np.linspace(-0.1, 0.1, 400, dtype=np.float32)
        np.testing.assert_array_equal(validate_audio(wave, 16000), wave)


class DecodingTests(unittest.TestCase):
    def test_merge_and_clip_partial_final_frame(self):
        segments = decode_scores([0.8, 0.9, 0.1, 0.7], 0.51)
        self.assertEqual(segments, [dict(start=0.0, end=0.32, score=0.9),
                                    dict(start=0.48, end=0.51, score=0.7)])

    def test_real_and_threshold_equality(self):
        self.assertEqual(decode_scores([0.1, 0.5], 0.32), [])
        self.assertEqual(decode_scores([0.1, 0.6], 0.32, threshold=0.7), [])

    def test_subframe_duration(self):
        self.assertEqual(decode_scores([0.9], 0.025),
                         [dict(start=0.0, end=0.025, score=0.9)])


if __name__ == '__main__':
    unittest.main()
