from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sal_runtime.detector import decode_scores, load_task_state, validate_audio


class SALContractTests(unittest.TestCase):
    def test_reject_wrong_input(self):
        for wave, rate in [(np.zeros(400, dtype=np.float32), 44100),
                           (np.zeros((400, 2)), 16000), (np.zeros(400, dtype=np.int16), 16000),
                           (np.full(400, np.nan), 16000), (np.zeros(399), 16000),
                           (np.zeros(340001), 16000)]:
            with self.subTest(shape=wave.shape, rate=rate), self.assertRaises(ValueError):
                validate_audio(wave, rate)

    def test_preserve_audio_amplitude(self):
        wave = np.linspace(-0.1, 0.1, 400, dtype=np.float32)
        np.testing.assert_array_equal(validate_audio(wave, 16000), wave)

    def test_merge_and_clip(self):
        self.assertEqual(decode_scores([0.9, 0.8, 0.1, 0.7], 0.51),
                         [dict(start=0.0, end=0.32, score=0.9),
                          dict(start=0.48, end=0.51, score=0.7)])

    def test_threshold_and_short_tail(self):
        self.assertEqual(decode_scores([0.5], 0.025), [])
        self.assertEqual(decode_scores([0.9], 0.025), [dict(start=0.0, end=0.025, score=0.9)])

    def test_incomplete_checkpoint_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'broken.ckpt'
            path.write_bytes(b'PK\x03\x04incomplete')
            with self.assertRaisesRegex(RuntimeError, 'Incomplete SAL checkpoint'):
                load_task_state(path)


if __name__ == '__main__':
    unittest.main()
