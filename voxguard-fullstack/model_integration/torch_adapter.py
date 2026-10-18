"""Configurable bridge for explicitly described PyTorch waveform models.

This module never guesses architecture, score semantics, or frame timing.
The supplied model source is trusted executable Python, not a data-only upload.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading

import numpy as np

from voxguard.adapters import AdapterError, AudioInput, Prediction


def _keys(value, name, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{name}: missing keys={sorted(missing)}; unknown keys={sorted(extra)}")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _lookup(value, path):
    if path in (None, ""):
        return value
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing dictionary key: {path}")
        value = value[key]
    return value


class TorchAdapter:
    def __init__(self, config_path):
        # Load and validate before describe() can advertise a ready model.
        import torch

        self.torch = torch
        self.path = Path(config_path).expanduser().resolve()
        self.config = json.loads(self.path.read_text(encoding="utf-8"))
        c = self.config
        _keys(c, "config", ("schema_version", "model_id", "display_name", "model_version", "source_revision", "is_mock",
                            "source_dir", "factory", "weights", "checkpoint_sha256", "input", "output", "thresholds", "localization"),
              ("model_kwargs", "state_dict_key", "device"))
        if type(c.get("schema_version")) is not int or c["schema_version"] != 1:
            raise ValueError("Unsupported model config schema_version")
        for name in ("model_id", "display_name", "model_version", "source_revision", "factory"):
            if not isinstance(c.get(name), str) or not c[name].strip():
                raise ValueError(f"{name} must be a nonempty string")
        if type(c.get("is_mock")) is not bool:
            raise ValueError("is_mock must be explicitly true or false")
        if c["is_mock"] != c["model_id"].startswith("example-"):
            raise ValueError("Mock IDs must start with example-; real IDs must not")
        self.input = c["input"]
        _keys(self.input, "input", ("sample_rate_hz", "channels", "layout", "min_duration_sec", "max_duration_sec"))
        self.rate = self.input["sample_rate_hz"]
        if type(self.rate) is not int or self.rate <= 0:
            raise ValueError("input.sample_rate_hz must be a positive integer")
        if self.input["layout"] not in ("BT", "BCT"):
            raise ValueError("input.layout must be BT or BCT")
        if self.input["channels"] not in ("mean", "preserve"):
            raise ValueError("input.channels must be mean or preserve")
        if self.input["layout"] == "BT" and self.input["channels"] != "mean":
            raise ValueError("BT requires channels=mean")
        lo = _positive(self.input["min_duration_sec"], "min_duration_sec")
        hi = _positive(self.input["max_duration_sec"], "max_duration_sec")
        if lo > hi:
            raise ValueError("min_duration_sec must not exceed max_duration_sec")
        self.output = c["output"]
        _keys(self.output, "output", ("global_key", "score_direction"), ("frame_key",))
        if self.output["score_direction"] not in ("higher_is_spoof", "higher_is_bonafide"):
            raise ValueError("Unknown output.score_direction")
        self.sign = 1.0 if self.output["score_direction"] == "higher_is_spoof" else -1.0
        for key in ("global_key", "frame_key"):
            value = self.output.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"output.{key} must be a string or null")
        _keys(c["thresholds"], "thresholds", ("detection", "localization", "version"))
        self.detection_threshold = _finite(c["thresholds"]["detection"], "detection threshold")
        if not isinstance(c["thresholds"].get("version"), str) or not c["thresholds"]["version"]:
            raise ValueError("thresholds.version is required")
        self.localization = c.get("localization")
        if self.localization is not None:
            loc = self.localization
            _keys(loc, "localization", ("stride_samples", "start_offset_samples", "frame_layout", "postprocessing_version"))
            stride = loc["stride_samples"]
            if type(stride) is not int or stride <= 0:
                raise ValueError("localization.stride_samples must be a positive integer")
            if type(loc.get("start_offset_samples")) is not int or loc["start_offset_samples"] != 0:
                raise ValueError("Generic bridge requires nonoverlapping bins beginning at sample 0")
            if loc.get("frame_layout") not in ("T", "BT"):
                raise ValueError("localization.frame_layout must be T or BT")
            if not isinstance(self.output.get("frame_key"), str) or not self.output["frame_key"]:
                raise ValueError("Localization requires output.frame_key")
            if not isinstance(loc.get("postprocessing_version"), str) or not loc["postprocessing_version"]:
                raise ValueError("localization.postprocessing_version is required")
            self.resolution = stride / self.rate
            self.localization_threshold = _finite(c["thresholds"]["localization"], "localization threshold")
        else:
            if c["thresholds"].get("localization") is not None:
                raise ValueError("Detection-only models require thresholds.localization=null")
            self.resolution = None
            self.localization_threshold = None

        weights = self._resolve(c["weights"])
        expected = c["checkpoint_sha256"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError("checkpoint_sha256 must contain the actual 64-character SHA-256")
        digest = hashlib.sha256()
        with weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected.lower():
            raise ValueError("Checkpoint SHA-256 does not match; model was not loaded")
        source = self._resolve(c["source_dir"])
        if not source.is_dir():
            raise ValueError("source_dir does not exist")
        sys.path.insert(0, str(source))
        module, sep, name = c["factory"].partition(":")
        if not sep or not module or not name:
            raise ValueError("factory must have the form module:create_model")
        factory = getattr(importlib.import_module(module), name)
        kwargs = c.get("model_kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError("model_kwargs must be an object")
        model = factory(**kwargs)
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Model factory must return torch.nn.Module")
        state = _lookup(torch.load(weights, map_location="cpu", weights_only=True), c.get("state_dict_key"))
        if not isinstance(state, dict) or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
            raise ValueError("Selected checkpoint object must be a tensor-only state_dict")
        model.load_state_dict(state, strict=True)
        self.device = torch.device(c.get("device", "cpu"))
        model.to(self.device)
        model.eval()
        self.model = model
        self.lock = threading.Lock()
        # Relative path names remain portable. Config changes or different weights
        # change the exposed version; source_revision binds the supplied source.
        fingerprint = hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()[:16]
        self.version = f'{c["model_version"]}-cfg{fingerprint}-w{digest.hexdigest()[:12]}'

    def _resolve(self, value):
        path = Path(value).expanduser()
        return (self.path.parent / path).resolve() if not path.is_absolute() else path.resolve()

    def describe(self):
        c = self.config
        return {"id": c["model_id"], "version": self.version, "display_name": c["display_name"],
                "status": "ready", "is_mock": c["is_mock"],
                "capabilities": {"detection": True, "localization": self.localization is not None},
                "input": {key: self.input[key] for key in ("sample_rate_hz", "min_duration_sec", "max_duration_sec")},
                "localization_resolution_sec": self.resolution,
                "thresholds": {"detection": self.detection_threshold, "localization": self.localization_threshold,
                               "version": c["thresholds"]["version"]}}

    def _prepare(self, audio):
        from scipy.signal import resample_poly

        wave = np.asarray(audio.waveform, dtype=np.float32)
        if wave.ndim != 2 or not wave.shape[0] or not wave.shape[1] or not np.isfinite(wave).all():
            raise AdapterError("INVALID_AUDIO", "模型输入必须是有限的 channels × samples 波形。")
        if type(audio.sample_rate_hz) is not int or audio.sample_rate_hz <= 0:
            raise AdapterError("INVALID_AUDIO", "模型输入采样率无效。")
        if not math.isfinite(audio.duration_sec) or abs(wave.shape[1] / audio.sample_rate_hz - audio.duration_sec) > 1 / audio.sample_rate_hz:
            raise AdapterError("INVALID_AUDIO", "音频时长与采样点数不一致。")
        if not self.input["min_duration_sec"] <= audio.duration_sec <= self.input["max_duration_sec"]:
            raise AdapterError("DURATION_OUT_OF_RANGE", "音频不符合所选模型的时长限制。")
        if self.input["channels"] == "mean":
            wave = wave.mean(axis=0, keepdims=True)
        if audio.sample_rate_hz != self.rate:
            divisor = math.gcd(audio.sample_rate_hz, self.rate)
            wave = resample_poly(wave, self.rate // divisor, audio.sample_rate_hz // divisor, axis=-1)
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        tensor = self.torch.from_numpy(wave)
        if self.input["layout"] == "BCT":
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _numpy(self, value):
        if self.torch.is_tensor(value):
            value = value.detach().cpu().float().numpy()
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("Model scores must be finite numbers")
        return array

    def predict(self, audio: AudioInput, mode: str, include_timeline: bool) -> Prediction:
        if mode not in ("detect", "detect_and_localize"):
            raise AdapterError("INVALID_REQUEST", "未知的检测模式。")
        if mode == "detect_and_localize" and self.localization is None:
            raise AdapterError("CAPABILITY_UNSUPPORTED", "当前模型未配置时间定位能力。")
        tensor = self._prepare(audio)
        try:
            with self.lock, self.torch.inference_mode():
                raw = self.model(tensor)
        except Exception as exc:
            raise AdapterError("INFERENCE_FAILED", "模型推理失败；请检查服务端模型日志。") from exc
        try:
            global_array = self._numpy(_lookup(raw, self.output.get("global_key")))
            if global_array.size != 1:
                raise ValueError("global_key must select exactly one score; adapt class logits explicitly")
            score = float(global_array.reshape(-1)[0]) * self.sign
            decision = {"label": "spoof" if score >= self.detection_threshold else "bonafide", "spoof_score": score,
                        "threshold": self.detection_threshold, "score_type": "uncalibrated", "score_direction": "higher_is_spoof"}
            loc = {"status": "not_requested" if self.localization else "unsupported", "segments": None,
                   "resolution_sec": None, "threshold": None, "postprocessing_version": None}
            timeline = None
            if mode == "detect_and_localize":
                frames = self._numpy(_lookup(raw, self.output["frame_key"]))
                if self.localization["frame_layout"] == "BT":
                    if frames.ndim != 2 or frames.shape[0] != 1:
                        raise ValueError("frame_layout=BT requires shape [1,T]")
                    frames = frames[0]
                elif frames.ndim != 1:
                    raise ValueError("frame_layout=T requires shape [T]")
                count = max(1, math.ceil(audio.duration_sec / self.resolution - 1e-9))
                if len(frames) != count:
                    raise ValueError(f"Full timeline coverage requires {count} bins; model returned {len(frames)}")
                timeline = [{"start_sec": i * self.resolution, "end_sec": min((i + 1) * self.resolution, audio.duration_sec),
                             "spoof_score": float(value) * self.sign} for i, value in enumerate(frames)]
                segments = []
                for item in timeline:
                    if item["spoof_score"] < self.localization_threshold:
                        continue
                    if segments and math.isclose(segments[-1]["end_sec"], item["start_sec"], abs_tol=1e-9):
                        previous = segments[-1]
                        first_duration = previous["end_sec"] - previous["start_sec"]
                        new_duration = item["end_sec"] - item["start_sec"]
                        previous["spoof_score"] = (previous["spoof_score"] * first_duration + item["spoof_score"] * new_duration) / (first_duration + new_duration)
                        previous["end_sec"] = item["end_sec"]
                    else:
                        segments.append(copy.copy(item))
                loc = {"status": "available", "segments": segments, "resolution_sec": self.resolution,
                       "threshold": self.localization_threshold, "postprocessing_version": self.localization["postprocessing_version"]}
            return Prediction({"decision": decision, "localization": loc, "timeline": timeline if include_timeline else None}, self.rate)
        except Exception as exc:
            raise AdapterError("INVALID_MODEL_OUTPUT", f"模型输出不符合配置：{exc}") from exc


def create_adapter():
    path = os.environ.get("VOXGUARD_MODEL_CONFIG")
    if not path:
        raise ValueError("Set VOXGUARD_MODEL_CONFIG to the model JSON config path")
    return TorchAdapter(path)
