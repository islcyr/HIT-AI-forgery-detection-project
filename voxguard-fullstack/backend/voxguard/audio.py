"""FFmpeg decoding fixes the original timeline before model preprocessing."""
import json
from pathlib import Path
import subprocess
import wave

import numpy as np

from .adapters import AdapterError, AudioInput


def run_media(args, **kwargs):
    try:
        return subprocess.run(args, check=True, timeout=30, stderr=subprocess.PIPE, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise AdapterError("INVALID_AUDIO", "音频解码超时，请检查文件。") from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        raise AdapterError("INVALID_AUDIO", "文件无法解码为有效音频。") from exc


def decode_audio(source: Path, task_dir: Path, audio_id: str, descriptor: dict, limits: dict):
    probe = run_media(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(source)], stdout=subprocess.PIPE)
    try:
        info = json.loads(probe.stdout)
        formats = set(info["format"]["format_name"].split(","))
        streams = [x for x in info["streams"] if x["codec_type"] == "audio"]
        if not formats.intersection({"wav", "flac", "mp3"}):
            raise AdapterError("UNSUPPORTED_FORMAT", "仅支持实际格式为 WAV、FLAC 或 MP3 的音频。")
        if len(streams) != 1:
            raise AdapterError("INVALID_AUDIO", "文件须包含一个音频流。")
        channels, rate = int(streams[0]["channels"]), int(streams[0]["sample_rate"])
    except (KeyError, ValueError, TypeError) as exc:
        raise AdapterError("INVALID_AUDIO", "音频元信息不完整。") from exc
    if channels not in (1, 2):
        raise AdapterError("CHANNELS_UNSUPPORTED", "仅支持单声道或双声道音频。")
    if rate <= 0 or rate > 384000:
        raise AdapterError("INVALID_AUDIO", "解码采样率无效或超过 384 kHz。")
    min_duration = max(limits["min_duration_sec"], descriptor["input"]["min_duration_sec"])
    max_duration = min(limits["max_duration_sec"], descriptor["input"]["max_duration_sec"])
    raw = task_dir / "decoded.f32"
    # Decoding one second beyond the accepted maximum detects oversized audio without
    # decoding an arbitrarily long compressed input. Oversized input always fails.
    run_media(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source), "-map", "0:a:0",
               "-t", str(max_duration + 1), "-f", "f32le", "-acodec", "pcm_f32le", str(raw)], stdout=subprocess.DEVNULL)
    samples = np.fromfile(raw, dtype="<f4")
    raw.unlink(missing_ok=True)
    if samples.size == 0 or samples.size % channels:
        raise AdapterError("NO_VALID_AUDIO", "文件没有有效音频采样。")
    waveform = samples.reshape(-1, channels).T.copy()
    duration = waveform.shape[1] / rate
    if not min_duration <= duration <= max_duration:
        raise AdapterError("DURATION_OUT_OF_RANGE", f"音频时长须在 {min_duration:g} 至 {max_duration:g} 秒之间。")
    if not np.isfinite(waveform).all():
        raise AdapterError("INVALID_AUDIO", "音频包含非有限采样值。")
    if not np.any(waveform):
        raise AdapterError("NO_VALID_AUDIO", "音频采样全部为零，无法进行有效分析。")
    # Playback is PCM16 for browser compatibility. Model input retains the original
    # float32 values, including decoder peaks outside [-1,1], without normalization.
    pcm = (np.clip(waveform.T, -1, 1 - 1 / 32768) * 32768).round().clip(-32768, 32767).astype("<i2")
    with wave.open(str(task_dir / "preview.wav"), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())
    return AudioInput(waveform, rate, duration, audio_id)


def resample_mono(audio: AudioInput, target_rate: int) -> np.ndarray:
    """Optional adapter helper: actual FFmpeg resampling, no trimming or gain normalization."""
    mono = np.mean(audio.waveform, axis=0, dtype=np.float32)
    if target_rate == audio.sample_rate_hz:
        return mono.copy()
    converted = run_media(["ffmpeg", "-v", "error", "-nostdin", "-f", "f32le", "-ar", str(audio.sample_rate_hz),
                           "-ac", "1", "-i", "pipe:0", "-f", "f32le", "-ar", str(target_rate), "-ac", "1", "pipe:1"],
                          input=mono.astype("<f4").tobytes(), stdout=subprocess.PIPE)
    return np.frombuffer(converted.stdout, dtype="<f4").copy()
