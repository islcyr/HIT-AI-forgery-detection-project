"""TFCL utterance detection with the original single-branch network."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import threading

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[3]
WINDOW_SAMPLES = 64600
AUXILIARY_PREFIXES = ('bidirectional_attn_T.', 'bidirectional_attn_D.', 'feature_proj_d.')
RNG_LOCK = threading.Lock()


@contextmanager
def inference_rng(seed, device):
    devices = [device.index or 0] if device.type == 'cuda' else []
    with RNG_LOCK, torch.random.fork_rng(devices=devices):
        numpy_state = np.random.get_state()
        try:
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
        finally:
            np.random.set_state(numpy_state)


def validate_audio(waveform, sample_rate):
    if sample_rate != 16000:
        raise ValueError('TFCL requires 16000 Hz audio; resample in the backend.')
    wave = np.asarray(waveform)
    if wave.ndim != 1 or not np.issubdtype(wave.dtype, np.floating):
        raise ValueError('TFCL requires a one-dimensional floating point mono waveform.')
    if not np.isfinite(wave).all():
        raise ValueError('Waveform contains NaN or infinity.')
    if not 400 <= len(wave) <= 340000:
        raise ValueError('TFCL deployment supports 400–340000 samples (0.025–21.25 s).')
    return np.ascontiguousarray(wave, dtype=np.float32)


def prepare_window(wave):
    """Exact official pad(): repeat short audio, then retain 64600 samples."""
    if len(wave) >= WINDOW_SAMPLES:
        return np.ascontiguousarray(wave[:WINDOW_SAMPLES], dtype=np.float32)
    return np.tile(wave, int(WINDOW_SAMPLES / len(wave)) + 1)[:WINDOW_SAMPLES].astype(np.float32)


class TFCLDetector:
    def __init__(self, device='auto', checkpoint=None, xlsr_checkpoint=None,
                 threshold=0.5, threads=4, seed=1234):
        if not 0 < threshold < 1 or threads < 1:
            raise ValueError('threshold must be between 0 and 1; threads must be positive.')
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu or enable GPU access.')
        self.threshold, self.seed = float(threshold), int(seed)
        torch.set_num_threads(threads)
        checkpoint = Path(checkpoint or ROOT / 'temp/weights/TFCL/TFCL_best_ckpt.pth')
        xlsr_checkpoint = Path(xlsr_checkpoint or ROOT / 'temp/weights/shared/xlsr2_300m.pt')
        for path in (checkpoint, xlsr_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {path}')
        from .upstream_model import Model
        self.model = Model(SimpleNamespace(), str(self.device), xlsr_checkpoint)
        # These training-only modules are absent from the released checkpoint
        # and are never traversed by forward(x, return_pair_loss=False).
        del self.model.bidirectional_attn_T
        del self.model.feature_proj_d
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        state = bundle.get('state_dict', bundle)
        state = {key: value for key, value in state.items() if not key.startswith(AUXILIARY_PREFIXES)}
        self.model.load_state_dict(state, strict=True)
        self.state_entry_count = len(state)
        self.model.to(self.device).eval()

    def predict_file(self, audio_path):
        wave, rate = sf.read(str(audio_path), dtype='float32')
        return self.predict(wave, rate)

    @torch.inference_mode()
    def predict(self, waveform, sample_rate):
        wave = validate_audio(waveform, sample_rate)
        windows = []
        # Upstream explicitly enables SSL train mode during eval (dropout and
        # layerdrop). Preserve that behavior, but fix/reset RNG per request so
        # repeated calls do not depend on earlier requests. Restore global RNG.
        with inference_rng(self.seed, self.device):
            for start in range(0, len(wave), WINDOW_SAMPLES):
                end = min(start + WINDOW_SAMPLES, len(wave))
                chunk = prepare_window(wave[start:end])
                logits = self.model(torch.from_numpy(chunk).unsqueeze(0).to(self.device))
                if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                    raise RuntimeError('Invalid TFCL output logits.')
                values = logits[0].cpu().tolist()
                score = float(logits.softmax(dim=-1)[0, 0].cpu())
                windows.append({'start': start / sample_rate, 'end': end / sample_rate,
                                'score': score, 'logits': values,
                                'raw_bonafide_score': values[1]})
        score = max(window['score'] for window in windows)
        fake = score > self.threshold
        return {
            'model': 'TFCL', 'checkpoint_variant': 'best',
            'sample_rate': int(sample_rate), 'duration': len(wave) / sample_rate,
            'is_fake': fake, 'label': 'fake' if fake else 'real', 'score': score,
            'score_type': 'max_window_softmax_fake_class_0_uncalibrated',
            'threshold': self.threshold, 'decision_rule': 'at_least_one_window_with_fake_score_gt_threshold',
            'supports_localization': False, 'segments': None,
            'window_seconds': WINDOW_SAMPLES / sample_rate, 'windows': windows,
            'ssl_mode': 'train_as_upstream', 'seed': self.seed,
        }
