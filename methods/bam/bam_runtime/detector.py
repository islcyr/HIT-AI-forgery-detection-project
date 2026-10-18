"""Single-utterance inference using the published BAM WavLM checkpoint."""
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[3]
RESOLUTION = 0.16
FRAME_SAMPLES = 2560
# Deployment guard for quadratic graph attention, not a paper/model limit.
MAX_SAMPLES = 340000


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('utils.tool', 'Attribution_Config'):
            return SimpleNamespace
        return super().find_class(module, name)


class _CheckpointPickle:
    __name__ = 'pickle'
    Unpickler = _CheckpointUnpickler


def validate_audio(waveform, sample_rate):
    if sample_rate != 16000:
        raise ValueError('BAM requires 16000 Hz audio; resample in the backend.')
    wave = np.asarray(waveform)
    if wave.ndim != 1 or not np.issubdtype(wave.dtype, np.floating):
        raise ValueError('BAM requires a one-dimensional floating point mono waveform.')
    if not np.isfinite(wave).all():
        raise ValueError('Waveform contains NaN or infinity.')
    if not 400 <= len(wave) <= MAX_SAMPLES:
        raise ValueError('BAM deployment supports 400–340000 samples (0.025–21.25 s); no truncation.')
    return np.ascontiguousarray(wave, dtype=np.float32)


def decode_scores(scores, duration, threshold=0.5):
    """Merge consecutive fake frames; clip the padded final frame to duration."""
    segments = []
    start = None
    for index in range(len(scores) + 1):
        positive = index < len(scores) and scores[index] > threshold
        if positive and start is None:
            start = index
        elif not positive and start is not None:
            segments.append({
                'start': start * RESOLUTION,
                'end': min(index * RESOLUTION, duration),
                'score': float(max(scores[start:index])),
            })
            start = None
    return segments


class BAMDetector:
    """Load once; predict(waveform, sample_rate) returns the common result fields."""
    def __init__(self, device='auto', checkpoint=None, wavlm_checkpoint=None,
                 threshold=0.5, threads=4):
        if not 0 < threshold < 1:
            raise ValueError('threshold must be strictly between 0 and 1.')
        if threads < 1:
            raise ValueError('threads must be positive.')
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --device cpu or enable GPU access.')
        torch.set_num_threads(threads)
        self.threshold = float(threshold)
        checkpoint = Path(checkpoint or ROOT / 'temp/weights/BAM/extracted/model.ckpt')
        wavlm_checkpoint = Path(wavlm_checkpoint or ROOT / 'temp/weights/shared/wavlm_large.pt')
        for path in (checkpoint, wavlm_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {path}; run scripts/setup_bam.sh.')
        # This is the trusted official Lightning checkpoint, with a custom config
        # class remapped to a namespace so training/Lightning imports are unnecessary.
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=False,
                            pickle_module=_CheckpointPickle)
        config = bundle['hyper_parameters']['config']
        args = bundle['hyper_parameters']['args']
        if config.ssl_name != 'wavlm_local' or args.resolution != RESOLUTION:
            raise ValueError('Expected the official BAM WavLM checkpoint at 0.16 s resolution.')
        config.ssl_ckpt = str(wavlm_checkpoint)
        from .vendor.bam import BAM
        self.model = BAM(args, config)
        state = bundle['state_dict']
        if not all(key.startswith('model.') for key in state):
            raise ValueError('Unexpected checkpoint state dictionary prefix.')
        self.model.load_state_dict({key[len('model.'):]: value for key, value in state.items()}, strict=True)
        del bundle, state
        self.model.to(self.device).eval()

    def predict_file(self, audio_path, include_frames=False):
        wave, rate = sf.read(str(audio_path), dtype='float32')
        return self.predict(wave, rate, include_frames=include_frames)

    @torch.inference_mode()
    def predict(self, waveform, sample_rate, include_frames=False):
        wave = validate_audio(waveform, sample_rate)
        duration = len(wave) / sample_rate
        # Match the official evaluation padding to label resolution. The model
        # itself then appends its original 256 samples before SSL extraction.
        frame_count = (len(wave) + FRAME_SAMPLES - 1) // FRAME_SAMPLES
        padded = np.pad(wave, (0, frame_count * FRAME_SAMPLES - len(wave)))
        logits, boundary = self.model(torch.from_numpy(padded).unsqueeze(0).to(self.device))
        if logits.shape != (1, frame_count, 2):
            raise RuntimeError(f'Unexpected BAM output shape: {tuple(logits.shape)}')
        # PartialSpoof labels: 0 = spoof, 1 = bona fide.
        scores = logits.softmax(dim=-1)[0, :, 0].cpu().tolist()
        if not np.isfinite(scores).all():
            raise RuntimeError('BAM produced non-finite scores.')
        segments = decode_scores(scores, duration, self.threshold)
        result = {
            'model': 'BAM', 'checkpoint_variant': 'WavLM-PartialSpoof',
            'sample_rate': int(sample_rate), 'duration': duration,
            'is_fake': bool(segments), 'label': 'fake' if segments else 'real',
            'score': float(max(scores)), 'score_type': 'max_frame_softmax_fake_class_0',
            'decision_rule': 'at_least_one_frame_with_fake_score_gt_threshold',
            'threshold': self.threshold, 'frame_resolution': RESOLUTION,
            'segments': segments,
        }
        if include_frames:
            result['frame_fake_scores'] = scores
            result['frame_score_type'] = 'softmax_fake_class_0_uncalibrated'
            result['frame_boundary_scores'] = boundary[0].cpu().tolist()
        return result
