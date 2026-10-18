"""PLUMBING FIXTURE ONLY: untrained energy transform, never a forgery detector."""
import torch
from torch import nn
import torch.nn.functional as F


class PlumbingFixture(nn.Module):
    def __init__(self, stride_samples=1600):
        super().__init__()
        self.stride_samples = stride_samples
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, waveform):
        if waveform.ndim == 3:
            waveform = waveform.mean(dim=1)
        size = waveform.shape[-1]
        pad = (-size) % self.stride_samples
        padded = F.pad(waveform.square(), (0, pad))
        frames = padded.unfold(-1, self.stride_samples, self.stride_samples).mean(dim=-1)
        # Correct the final mean for zero padding; timeline represents original audio.
        valid = size % self.stride_samples
        if valid:
            frames = frames.clone()
            frames[..., -1] *= self.stride_samples / valid
        return {"global_score": waveform.square().mean(dim=-1) * self.scale + self.bias,
                "frame_scores": frames * self.scale + self.bias}


def create_model(**kwargs):
    return PlumbingFixture(**kwargs)
