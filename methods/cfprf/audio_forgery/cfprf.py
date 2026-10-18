"""CFPRF adapter: original FDN, PRN, and proposal decoding, without labels."""
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'temp' / 'CFPRF'
sys.path.insert(0, str(SOURCE))
from models.FDN import CFPRF_FDN
from models.PRN import CFPRF_PRN
from libs.tool import segscore2proposal, proposal_func, decoder_reg


def validate_audio(waveform, sample_rate):
    """Validate the backend contract; do not normalize or resample audio."""
    if sample_rate != 16000:
        raise ValueError('CFPRF requires 16000 Hz audio; resample in the backend.')
    wave = np.asarray(waveform)
    if wave.ndim != 1:
        raise ValueError('CFPRF requires a one-dimensional mono waveform.')
    if not np.issubdtype(wave.dtype, np.floating):
        raise ValueError('Waveform must be floating point PCM, normally in [-1, 1].')
    if not np.isfinite(wave).all():
        raise ValueError('Waveform contains NaN or infinity.')
    if not 400 <= len(wave) <= 340000:
        raise ValueError('CFPRF supports 400–340000 samples (0.025–21.25 s) per call; audio is never silently truncated.')
    return np.ascontiguousarray(wave, dtype=np.float32)


def refine_proposals(prn, features, coarse, threshold=0.5):
    """Original verification + regression decoding for a single utterance."""
    if not len(coarse):
        return np.empty((0, 3), dtype=np.float64)
    verification, regression, _, proposal_list = prn(features, [coarse], 20)
    proposals = proposal_list[0]
    positive = (verification.detach().cpu().numpy().reshape(-1) > threshold)
    refined = decoder_reg(proposals, regression.detach().cpu().numpy())
    # Match the original evaluator: skip regression for proposals < 2 frames,
    # and revert to the coarse interval if the regressed end becomes zero.
    short = (proposals[:, 2] - proposals[:, 1]) < 2
    fallback = short | (refined[:, 2] == 0)
    refined[fallback] = proposals[fallback]
    return refined[positive]


class CFPRFDetector:
    """Load once, then call predict(waveform, sample_rate) for each audio."""
    def __init__(self, variant='HAD', device='auto', weights_dir=None, xlsr_checkpoint=None, threads=4):
        variant = variant.upper()
        if variant not in ('HAD', 'PS', 'LAVDF'):
            raise ValueError('variant must be HAD, PS, or LAVDF')
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this process. Check GPU device access and the driver.')
        if threads < 1:
            raise ValueError('threads must be positive')
        torch.set_num_threads(threads)
        self.variant = variant
        weights_dir = Path(weights_dir) if weights_dir else ROOT / 'temp' / 'weights' / 'CFPRF'
        xlsr_checkpoint = Path(xlsr_checkpoint) if xlsr_checkpoint else ROOT / 'temp' / 'weights' / 'shared' / 'xlsr2_300m.pt'
        paths = [xlsr_checkpoint, weights_dir / f'1FDN_{variant}.pth', weights_dir / f'2PRN_{variant}.pth']
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.fdn = CFPRF_FDN(xlsr_checkpoint=str(xlsr_checkpoint), device=self.device)
        # Task checkpoints are plain state dicts. Strict loading catches a wrong
        # architecture, checkpoint variant, or missing trained parameters.
        self.fdn.load_state_dict(torch.load(paths[1], map_location='cpu', weights_only=True), strict=True)
        self.fdn.to(self.device).eval()
        self.prn = CFPRF_PRN(device=self.device)
        self.prn.load_state_dict(torch.load(paths[2], map_location='cpu', weights_only=True), strict=True)
        self.prn.to(self.device).eval()

    def predict_file(self, audio_path, include_frames=False):
        wave, rate = sf.read(str(audio_path), dtype='float32')
        return self.predict(wave, rate, include_frames=include_frames)

    @torch.inference_mode()
    def predict(self, waveform, sample_rate, include_frames=False):
        wave = validate_audio(waveform, sample_rate)
        duration = float(len(wave) / sample_rate)
        scores, _, _, features = self.fdn(torch.from_numpy(wave).unsqueeze(0).to(self.device))
        logits = scores[0].detach().cpu().numpy()
        coarse = segscore2proposal(logits, cp_fun=proposal_func, rso=20)[1]
        refined = refine_proposals(self.prn, features, coarse)
        segments = []
        for confidence, start_frame, end_frame in refined:
            start = float(np.clip(start_frame * 0.02, 0, duration))
            end = float(np.clip(end_frame * 0.02, 0, duration))
            if np.isfinite([confidence, start, end]).all() and end > start:
                segments.append({'start':start, 'end':end, 'score':float(confidence)})
        segments.sort(key=lambda segment: (segment['start'], segment['end']))
        result = {
            'model':'CFPRF', 'checkpoint_variant':self.variant,
            'sample_rate':int(sample_rate), 'duration':duration,
            'is_fake':bool(segments), 'label':'fake' if segments else 'real',
            'score':max((segment['score'] for segment in segments), default=0.0),
            'score_type':'max_refined_proposal_verification',
            'decision_rule':'at_least_one_refined_proposal_with_verification_gt_0.5',
            'frame_resolution':0.02, 'segments':segments,
        }
        if include_frames:
            # Original labels: class 0 = fake, class 1 = real.
            result['frame_fake_scores'] = torch.softmax(scores[0], dim=-1)[:, 0].cpu().tolist()
            result['frame_score_type'] = 'softmax_fake_class_0_uncalibrated'
        return result
