"""Configurable command runner and native-output converter, no model imports."""
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from voxguard.adapters import AdapterError, Prediction
from voxguard.audio import resample_mono
from voxguard.schemas import Descriptor, validate_prediction


class CommandAdapter:
    def __init__(self, config, base_dir):
        self.config = config
        self.descriptor = Descriptor.model_validate(config["descriptor"])
        self.cwd = (base_dir / config["cwd"]).resolve()
        if not self.cwd.is_dir():
            raise ValueError(f"Model project directory does not exist: {self.cwd}")
        self.command = config["command"]
        if not isinstance(self.command, list) or not self.command or not all(isinstance(x, str) for x in self.command):
            raise ValueError("command must be a nonempty argument list")
        if not any("{audio}" in x for x in self.command) or not any("{output}" in x for x in self.command):
            raise ValueError("command must include {audio} and {output}")
        self.timeout = float(config.get("timeout_sec", 540))
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout_sec must be positive")
        module, sep, name = config["output_converter"].partition(":")
        if not sep or not module or not name:
            raise ValueError("output_converter must be module:function")
        self.converter = getattr(importlib.import_module(module), name)
        if not callable(self.converter):
            raise ValueError("output_converter must be callable")
        # These are prerequisite files, not an assertion of CUDA/model readiness.
        for filename in config.get("required_files", []):
            if not (self.cwd / filename).is_file():
                raise ValueError(f"Required model file does not exist: {filename}")

    def describe(self):
        return self.descriptor.model_dump()

    def predict(self, audio, mode, include_timeline):
        rate = self.descriptor.input.sample_rate_hz
        waveform = resample_mono(audio, rate)
        # IEEE float WAV preserves amplitudes, including values outside PCM16 range.
        import struct
        samples = np.asarray(waveform, dtype='<f4').tobytes()
        with tempfile.TemporaryDirectory(prefix="voxguard-model-") as temp:
            source, output = Path(temp) / "input.wav", Path(temp) / "result.json"
            source.write_bytes(b'RIFF' + struct.pack('<I', 36 + len(samples)) + b'WAVEfmt ' +
                               struct.pack('<IHHIIHH', 16, 3, 1, rate, rate * 4, 4, 32) +
                               b'data' + struct.pack('<I', len(samples)) + samples)
            command = [arg.replace("{audio}", str(source)).replace("{output}", str(output)) for arg in self.command]
            # Inherit worker log descriptors rather than buffering unbounded model logs.
            try:
                subprocess.run(command, cwd=self.cwd, check=True, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                logging.exception("Model command timed out")
                raise AdapterError("MODEL_TIMEOUT", "模型命令执行超时。", True) from exc
            except (OSError, subprocess.CalledProcessError) as exc:
                logging.exception("Model command failed")
                raise AdapterError("INFERENCE_FAILED", "模型命令执行失败，请查看任务日志。") from exc
            try:
                raw = json.loads(output.read_text(encoding="utf-8"))
                context = {**self.config.get("converter_options", {}), **self.describe()}
                prediction = self.converter(raw, audio, context, mode)
                if not isinstance(prediction, Prediction) or prediction.analysis_sample_rate_hz != rate:
                    raise ValueError("Converter must return Prediction with configured sample rate")
                validate_prediction(prediction.result, self.descriptor, audio.duration_sec, mode)
                return prediction
            except (OSError, ValueError, TypeError, KeyError) as exc:
                logging.exception("Native model output conversion failed")
                raise AdapterError("INVALID_MODEL_OUTPUT", "模型原始输出不符合转换约定，请查看任务日志。") from exc


def create_adapter():
    """A single factory can register multiple command-based models."""
    path = Path(os.environ["VOXGUARD_COMMAND_CONFIG"]).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    entries = config.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Configuration requires a nonempty models list")
    return [CommandAdapter(entry, path.parent) for entry in entries]
