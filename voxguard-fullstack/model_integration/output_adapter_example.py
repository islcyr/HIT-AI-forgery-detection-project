"""MOCK ONLY: executable example of adapting a model's tuple/logit output.

The energy-based network below is untrained and has no forgery-detection ability.
Its explicitly defined convention is: a larger logit means more bonafide.
The wrapper converts both heads to higher-is-spoof scores using 1 - sigmoid.
Those numbers are NOT calibrated probabilities. The unrelated auxiliary output
is not used as a prediction. This file can be copied into an isolated source_dir.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _validate_stride(stride_samples):
    if type(stride_samples) is not int or stride_samples <= 0:
        raise ValueError("stride_samples must be a positive integer")


def _validate_waveform(waveform):
    if not torch.is_tensor(waveform) or not waveform.is_floating_point():
        raise ValueError("waveform must be a floating-point tensor")
    if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
        raise ValueError("waveform must have nonempty shape [batch, samples]")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform must contain only finite values")


class MockTupleNetwork(nn.Module):
    """Untrained mock with separate frame and utterance logit outputs.

    Every input sample contributes to a frame, including a short final frame.
    Padding is an internal batching convenience: its zeros are excluded from
    the energy mean. No tail is discarded or assigned a copied frame score.
    """

    is_mock = True

    def __init__(self, stride_samples=1600):
        super().__init__()
        _validate_stride(stride_samples)
        self.stride_samples = stride_samples
        self.frame_scale = nn.Parameter(torch.tensor(2.0))
        self.frame_bias = nn.Parameter(torch.tensor(-0.4))
        self.utterance_scale = nn.Parameter(torch.tensor(3.0))
        self.utterance_bias = nn.Parameter(torch.tensor(-0.3))

    def forward(self, waveform):
        _validate_waveform(waveform)
        samples = waveform.shape[1]
        frame_count = (samples + self.stride_samples - 1) // self.stride_samples
        power = waveform.square()
        padded = F.pad(power, (0, frame_count * self.stride_samples - samples))
        frame_power_sum = padded.reshape(waveform.shape[0], frame_count, self.stride_samples).sum(dim=-1)
        counts = torch.full((frame_count,), self.stride_samples, device=waveform.device, dtype=waveform.dtype)
        counts[-1] = samples - (frame_count - 1) * self.stride_samples
        frame_energy = frame_power_sum / counts
        frame_logits = frame_energy * self.frame_scale + self.frame_bias
        utterance_logits = power.mean(dim=-1, keepdim=True) * self.utterance_scale + self.utterance_bias
        auxiliary = {"frame_energy": frame_energy, "valid_frame_samples": counts}
        return frame_logits, utterance_logits, auxiliary


class TupleOutputAdapter(nn.Module):
    """Normalize this example's known tuple shapes and score direction.

    This is an output wrapper for TorchAdapter's existing model factory, not a
    replacement HTTP API adapter. Different architectures may need different
    shape rules, class selection, preprocessing or time mapping.
    """

    is_mock = True

    def __init__(self, network=None, stride_samples=1600):
        super().__init__()
        _validate_stride(stride_samples)
        self.stride_samples = stride_samples
        self.network = MockTupleNetwork(stride_samples) if network is None else network
        if not isinstance(self.network, nn.Module):
            raise ValueError("network must be torch.nn.Module")

    def forward(self, waveform):
        _validate_waveform(waveform)
        raw = self.network(waveform)
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise ValueError("model must return (frame_logits, utterance_logits, auxiliary)")
        frame_logits, utterance_logits, _auxiliary = raw
        batch, samples = waveform.shape
        frame_count = (samples + self.stride_samples - 1) // self.stride_samples
        for name, value, expected in (
            ("frame_logits", frame_logits, (batch, frame_count)),
            ("utterance_logits", utterance_logits, (batch, 1)),
        ):
            if not torch.is_tensor(value) or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating-point tensor")
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}; received {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        # No additional sign inversion is configured in the generic bridge.
        frame_scores = 1.0 - torch.sigmoid(frame_logits)
        global_score = (1.0 - torch.sigmoid(utterance_logits)).squeeze(-1)
        if not torch.isfinite(frame_scores).all() or not torch.isfinite(global_score).all():
            raise ValueError("converted model scores must contain only finite values")
        return {"global_score": global_score, "frame_scores": frame_scores}


def create_model(stride_samples=1600):
    """Factory used by generated config; the generic bridge loads all weights."""
    return TupleOutputAdapter(stride_samples=stride_samples)
