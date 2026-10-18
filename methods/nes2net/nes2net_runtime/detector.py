"""Nes2Net-X single-file inference, preserving upstream raw scoring."""
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[3]


def validate_audio(waveform, sample_rate):
    if sample_rate != 16000:
        raise ValueError('Nes2Net requires 16000 Hz audio; resample in the backend.')
    wave = np.asarray(waveform)
    if wave.ndim != 1 or not np.issubdtype(wave.dtype, np.floating):
        raise ValueError('Nes2Net requires a one-dimensional floating point mono waveform.')
    if not np.isfinite(wave).all():
        raise ValueError('Waveform contains NaN or infinity.')
    if not 400 <= len(wave) <= 340000:
        raise ValueError('Nes2Net deployment supports 400–340000 samples (0.025–21.25 s).')
    return np.ascontiguousarray(wave, dtype=np.float32)


def prepare_audio(wave, test_mode):
    if test_mode == 'full':
        return wave
    if test_mode != '4s':
        raise ValueError('test_mode must be full or 4s.')
    # Official easy_inference_demo.py uses 64000, not the batch loader's 64600.
    if len(wave) >= 64000:
        return wave[:64000]
    return np.tile(wave, int(64000 / len(wave)) + 1)[:64000]


def make_result(logits, sample_count, processed_count, test_mode, threshold):
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise RuntimeError('Invalid Nes2Net output logits.')
    values = logits[0].detach().cpu().tolist()
    real_score = float(values[1])
    fake = real_score < threshold if threshold is not None else None
    return {
        'model': 'Nes2Net-X', 'checkpoint_variant': 'DF-best-single',
        'sample_rate': 16000, 'duration': sample_count / 16000,
        'score': real_score, 'score_type': 'raw_bonafide_logit',
        'score_direction': 'higher_is_real', 'logits': values,
        'fake_softmax_score': float(logits.softmax(-1)[0, 0].detach().cpu()),
        'is_fake': fake, 'label': 'unclassified' if fake is None else 'fake' if fake else 'real',
        'threshold': threshold,
        'decision_rule': 'score_only_no_calibrated_threshold' if threshold is None else 'fake_if_raw_bonafide_logit_lt_threshold',
        'supports_localization': False, 'segments': None,
        'test_mode': test_mode, 'processed_samples': processed_count,
        'evaluated_end_seconds': min(sample_count, processed_count) / 16000,
    }


class Nes2NetDetector:
    def __init__(self, device='auto', checkpoint=None, xlsr_checkpoint=None,
                 test_mode='full', threshold=None, threads=4):
        if test_mode not in ('full', '4s') or threads < 1:
            raise ValueError('test_mode must be full or 4s; threads must be positive.')
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError('threshold must be finite, in raw bonafide logit units.')
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu or enable GPU access.')
        self.test_mode, self.threshold = test_mode, threshold
        torch.set_num_threads(threads)
        checkpoint = Path(checkpoint or ROOT / 'temp/weights/Nes2Net/Nes2Net-X_DF_best.pth')
        xlsr_checkpoint = Path(xlsr_checkpoint or ROOT / 'temp/weights/shared/xlsr2_300m.pt')
        for path in (checkpoint, xlsr_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {path}')
        from .upstream_model import wav2vec2_Nes2Net_no_Res_w_allT
        args = SimpleNamespace(n_output_logits=2, Nes_ratio=[8, 8], SE_ratio=[1],
                               dilation=2, pool_func='mean')
        self.model = wav2vec2_Nes2Net_no_Res_w_allT(args, str(self.device), xlsr_checkpoint)
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.state_entry_count = len(state)
        self.model.to(self.device).eval()

    def predict_file(self, audio_path):
        wave, rate = sf.read(str(audio_path), dtype='float32')
        return self.predict(wave, rate)

    @torch.inference_mode()
    def predict(self, waveform, sample_rate):
        wave = validate_audio(waveform, sample_rate)
        prepared = prepare_audio(wave, self.test_mode)
        logits = self.model(torch.from_numpy(prepared).unsqueeze(0).to(self.device))
        return make_result(logits, len(wave), len(prepared), self.test_mode, self.threshold)
