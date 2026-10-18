"""Label-free single-audio SAL inference with the author's network and weights."""
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[3]
RESOLUTION = 0.16
FRAME_SAMPLES = 2560
VARIANTS = {'W2V2': ('xlsr', 'xlsr2_300m.pt'), 'WavLM': ('large', 'wavlm_large.pt')}


def validate_audio(waveform, sample_rate):
    if sample_rate != 16000:
        raise ValueError('SAL requires 16000 Hz audio; resample in the backend.')
    wave = np.asarray(waveform)
    if wave.ndim != 1 or not np.issubdtype(wave.dtype, np.floating):
        raise ValueError('SAL requires a one-dimensional floating point mono waveform.')
    if not np.isfinite(wave).all():
        raise ValueError('Waveform contains NaN or infinity.')
    if not 400 <= len(wave) <= 340000:
        raise ValueError('SAL deployment supports 400–340000 samples (0.025–21.25 s); no truncation.')
    return np.ascontiguousarray(wave, dtype=np.float32)


def decode_scores(scores, duration, threshold=0.5):
    segments = []
    start = None
    for index in range(len(scores) + 1):
        positive = index < len(scores) and scores[index] > threshold
        if positive and start is None:
            start = index
        elif not positive and start is not None:
            segments.append({'start': start * RESOLUTION,
                             'end': min(index * RESOLUTION, duration),
                             'score': float(max(scores[start:index]))})
            start = None
    return segments


def load_task_state(checkpoint):
    """Reject incomplete archives before model allocation; never fill missing weights."""
    try:
        with zipfile.ZipFile(checkpoint):
            pass
    except zipfile.BadZipFile as error:
        raise RuntimeError(
            f'Incomplete SAL checkpoint: {checkpoint}. Missing ZIP archive index; '
            'the downloaded official SAL_W2V2.ckpt is truncated. '
            'Supply a complete checkpoint with --checkpoint; missing trained weights '
            'will not be replaced with random or base-model parameters.'
        ) from error
    bundle = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = bundle.get('state_dict', bundle)
    if not state or not all(key.startswith('net.') for key in state):
        raise ValueError('Expected official SAL state_dict with net. prefix.')
    return {key[4:]: value for key, value in state.items()}


class SALDetector:
    """Load once; predict accepts one preprocessed waveform and returns common fields."""
    def __init__(self, variant='WavLM', device='auto', checkpoint=None,
                 encoder_checkpoint=None, threshold=0.5, threads=4):
        if variant not in VARIANTS:
            raise ValueError('variant must be W2V2 or WavLM.')
        if not 0 < threshold < 1 or threads < 1:
            raise ValueError('threshold must be between 0 and 1; threads must be positive.')
        self.variant = variant
        self.threshold = float(threshold)
        checkpoint = Path(checkpoint or ROOT / f'temp/weights/SAL/SAL_{variant}.ckpt')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing SAL task checkpoint: {checkpoint}')
        state = load_task_state(checkpoint)
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --device cpu or enable GPU access.')
        torch.set_num_threads(threads)
        encoder_name, encoder_file = VARIANTS[variant]
        encoder_checkpoint = Path(encoder_checkpoint or ROOT / 'temp/weights/shared' / encoder_file)
        if not encoder_checkpoint.is_file():
            raise FileNotFoundError(f'Missing SSL encoder checkpoint: {encoder_checkpoint}')
        from .vendor.net.model import SSLSeq8Bin
        self.model = SSLSeq8Bin(
            ssl_encoder=encoder_name, ckpt=str(encoder_checkpoint), mode='s3prl_weighted',
            hid_dim=1024, seq_model='cf', num_layers=2, num_heads=4,
            resolution_train=RESOLUTION, resolution_test=RESOLUTION, pool='avg')
        self.model.load_state_dict(state, strict=True)
        self.state_entry_count = len(state)
        del state
        self.model.to(self.device).eval()

    def predict_file(self, audio_path, include_frames=False):
        wave, rate = sf.read(str(audio_path), dtype='float32')
        return self.predict(wave, rate, include_frames=include_frames)

    @torch.inference_mode()
    def predict(self, waveform, sample_rate, include_frames=False):
        wave = validate_audio(waveform, sample_rate)
        duration = len(wave) / sample_rate
        frames = (len(wave) + FRAME_SAMPLES - 1) // FRAME_SAMPLES
        # Official test data pipeline aligns waveform length to 0.16 s labels.
        padded = np.pad(wave, (0, frames * FRAME_SAMPLES - len(wave)))
        positions, logits = self.model(torch.from_numpy(padded).unsqueeze(0).to(self.device))
        if logits.shape != (1, frames, 2) or positions.shape != (1, frames, 8):
            raise RuntimeError(f'Unexpected SAL output shapes: {logits.shape}, {positions.shape}')
        # Official evaluation uses the binary head; class 0 = fake, 1 = real.
        scores = logits.softmax(dim=-1)[0, :, 0].cpu().tolist()
        if not np.isfinite(scores).all():
            raise RuntimeError('SAL produced non-finite scores.')
        segments = decode_scores(scores, duration, self.threshold)
        result = {
            'model': 'SAL', 'checkpoint_variant': self.variant,
            'sample_rate': int(sample_rate), 'duration': duration,
            'is_fake': bool(segments), 'label': 'fake' if segments else 'real',
            'score': float(max(scores)), 'score_type': 'max_frame_softmax_fake_class_0',
            'decision_rule': 'at_least_one_frame_with_fake_score_gt_threshold',
            'threshold': self.threshold, 'frame_resolution': RESOLUTION, 'segments': segments,
        }
        if include_frames:
            result['frame_fake_scores'] = scores
            result['frame_score_type'] = 'softmax_fake_class_0_uncalibrated'
        return result
