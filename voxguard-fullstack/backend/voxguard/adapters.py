"""Public plugin boundary. No sample labels are accessible through this interface."""
from dataclasses import dataclass
import importlib
from typing import Protocol

import numpy as np

from .schemas import Descriptor


@dataclass(frozen=True)
class AudioInput:
    waveform: np.ndarray
    sample_rate_hz: int
    duration_sec: float
    audio_id: str


@dataclass(frozen=True)
class Prediction:
    result: dict
    analysis_sample_rate_hz: int


class AdapterError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable

    def detail(self):
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


class Adapter(Protocol):
    def describe(self) -> dict: ...
    def predict(self, audio: AudioInput, mode: str, include_timeline: bool) -> Prediction: ...


class MockAdapter:
    """Fixed synthetic output exercises the UI; it never detects audio authenticity."""
    def __init__(self, localization=True):
        self.localization = localization

    def describe(self):
        return {"id": "example-localizer" if self.localization else "example-detector", "version": "example-v0.2",
                "display_name": "模拟定位模型（非真实检测）" if self.localization else "模拟整段模型（非真实检测）",
                "status": "ready", "is_mock": True,
                "capabilities": {"detection": True, "localization": self.localization},
                "input": {"sample_rate_hz": 16000, "min_duration_sec": 0.5, "max_duration_sec": 120.0},
                "localization_resolution_sec": 0.16 if self.localization else None,
                "thresholds": {"detection": 0.8, "localization": 0.8 if self.localization else None, "version": "example-threshold-v1"}}

    def predict(self, audio, mode, include_timeline):
        # The mock has no learned weights. Actual sample conversion still exercises preprocessing.
        from .audio import resample_mono
        resample_mono(audio, 16000)
        decision = {"label": "spoof", "spoof_score": 1.6, "threshold": 0.8,
                    "score_type": "uncalibrated", "score_direction": "higher_is_spoof"}
        loc = {"status": "not_requested" if self.localization else "unsupported", "segments": None,
               "resolution_sec": None, "threshold": None, "postprocessing_version": None}
        timeline = None
        if mode == "detect_and_localize":
            timeline = []
            index = 0
            while index * 0.16 < audio.duration_sec:
                start = index * 0.16
                end = min((index + 1) * 0.16, audio.duration_sec)
                score = 1.6 if (index // 4) % 3 == 1 else -0.4
                timeline.append({"start_sec": start, "end_sec": end, "spoof_score": score})
                index += 1
            segments = []
            for item in timeline:
                if item["spoof_score"] >= 0.8:
                    if segments and segments[-1]["end_sec"] == item["start_sec"]:
                        segments[-1]["end_sec"] = item["end_sec"]
                    else:
                        segments.append(dict(item))
            loc = {"status": "available", "segments": segments, "resolution_sec": 0.16,
                   "threshold": 0.8, "postprocessing_version": "example-fixed-pattern-v1"}
        return Prediction({"decision": decision, "localization": loc, "timeline": timeline}, 16000)


def load_adapters(enable_mock: bool, factory_path: str):
    adapters = [MockAdapter(True), MockAdapter(False)] if enable_mock else []
    if factory_path:
        module, separator, name = factory_path.partition(":")
        if not separator or not module or not name:
            raise ValueError("VOXGUARD_ADAPTER_FACTORY must have the form module:function")
        created = getattr(importlib.import_module(module), name)()
        plugins = created if isinstance(created, (list, tuple)) else [created]
        adapters.extend(plugins)
    registry = {}
    for adapter in adapters:
        descriptor = Descriptor.model_validate(adapter.describe())
        if descriptor.id in registry:
            raise ValueError(f"Duplicate model id: {descriptor.id}")
        if descriptor.status == "ready":
            registry[descriptor.id] = (adapter, descriptor)
    return registry
