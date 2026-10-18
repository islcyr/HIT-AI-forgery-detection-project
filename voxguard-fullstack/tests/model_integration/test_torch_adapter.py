import hashlib
import json

import numpy as np
import pytest
import torch

from model_integration.make_fixture import make_fixture
from model_integration.torch_adapter import TorchAdapter
from voxguard.adapters import AdapterError, AudioInput
from voxguard.schemas import Descriptor, validate_prediction


@pytest.fixture
def config_path(tmp_path):
    return make_fixture(tmp_path / "model")


def modify(path, callback):
    config = json.loads(path.read_text())
    callback(config)
    path.write_text(json.dumps(config), encoding="utf-8")


def audio(rate=16000):
    wave = np.zeros((1, int(rate * .65)), dtype=np.float32)
    wave[:, :int(rate * .2)] = 1
    wave[:, int(rate * .4):] = .8
    return AudioInput(wave, rate, wave.shape[-1] / rate, "test-audio")


def test_actual_weights_inference_timing_and_unequal_last_bin(config_path):
    adapter = TorchAdapter(config_path)
    assert not adapter.model.training
    assert adapter.model.scale.item() == pytest.approx(1.25)
    assert adapter.model.bias.item() == pytest.approx(-.1)
    original = adapter.model.forward
    modes = []

    def forward(wave):
        modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return original(wave)

    adapter.model.forward = forward
    source = audio()
    prediction = adapter.predict(source, "detect_and_localize", True)
    assert modes == [(False, True)]
    assert adapter.describe()["is_mock"] is True
    assert adapter.describe()["id"] == "example-plumbing-fixture"
    assert prediction.analysis_sample_rate_hz == 16000
    assert prediction.result["decision"]["spoof_score"] == pytest.approx(((.2 + .25 * .64) / .65) * 1.25 - .1)
    bins = prediction.result["timeline"]
    assert len(bins) == 7
    assert bins[-1]["start_sec"] == pytest.approx(.6)
    assert bins[-1]["end_sec"] == pytest.approx(.65)
    segments = prediction.result["localization"]["segments"]
    assert len(segments) == 2
    assert segments[0]["start_sec"] == 0
    assert segments[0]["end_sec"] == pytest.approx(.2)
    assert segments[1]["start_sec"] == pytest.approx(.4)
    assert segments[1]["end_sec"] == pytest.approx(.65)
    assert segments[1]["spoof_score"] == pytest.approx(.7)
    validate_prediction(prediction.result, Descriptor.model_validate(adapter.describe()), source.duration_sec, "detect_and_localize")


def test_real_resampling_and_stereo_mean(config_path):
    adapter = TorchAdapter(config_path)
    source = audio(44100)
    source = AudioInput(np.concatenate([source.waveform, -source.waveform]), source.sample_rate_hz, source.duration_sec, "stereo")
    prediction = adapter.predict(source, "detect_and_localize", True)
    assert prediction.result["decision"]["spoof_score"] == pytest.approx(-.1)
    assert prediction.result["localization"]["segments"] == []
    assert len(prediction.result["timeline"]) == 7


def test_score_direction_thresholds_and_version_are_bound(config_path):
    first = TorchAdapter(config_path)
    modify(config_path, lambda c: (c["output"].update(score_direction="higher_is_bonafide"), c["thresholds"].update(detection=-.5, localization=-.2)))
    second = TorchAdapter(config_path)
    assert first.version != second.version
    result = second.predict(audio(), "detect_and_localize", True).result
    assert result["decision"]["spoof_score"] < 0
    assert result["decision"]["label"] == "bonafide"
    assert result["decision"]["threshold"] == -.5
    assert len(result["localization"]["segments"]) == 1
    assert result["localization"]["segments"][0]["start_sec"] == pytest.approx(.2)
    assert result["localization"]["segments"][0]["end_sec"] == pytest.approx(.4)


def test_checkpoint_checksum_checked_before_ready(config_path):
    modify(config_path, lambda c: c.update(checkpoint_sha256="0" * 64))
    with pytest.raises(ValueError, match="SHA-256"):
        TorchAdapter(config_path)


def test_missing_checkpoint_fails_at_startup(config_path):
    modify(config_path, lambda c: c.update(weights="missing.pt"))
    with pytest.raises(FileNotFoundError):
        TorchAdapter(config_path)


def test_state_dict_loaded_strictly(config_path):
    weights = config_path.parent / "plumbing_fixture.pt"
    torch.save({"state_dict": {"scale": torch.tensor(1.25)}}, weights)
    modify(config_path, lambda c: c.update(checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest()))
    with pytest.raises(RuntimeError, match="Missing key"):
        TorchAdapter(config_path)


def test_wrong_frame_count_does_not_fabricate_coverage(config_path):
    modify(config_path, lambda c: c["localization"].update(stride_samples=800))
    adapter = TorchAdapter(config_path)
    with pytest.raises(AdapterError) as err:
        adapter.predict(audio(), "detect_and_localize", True)
    assert err.value.code == "INVALID_MODEL_OUTPUT"
    assert "13 bins" in str(err.value)


def test_missing_outputs_and_nonfinite_scores_are_rejected(config_path):
    adapter = TorchAdapter(config_path)
    adapter.model.forward = lambda wave: {"global_score": torch.tensor(float("nan")), "frame_scores": torch.zeros(1, 7)}
    with pytest.raises(AdapterError) as err:
        adapter.predict(audio(), "detect_and_localize", True)
    assert err.value.code == "INVALID_MODEL_OUTPUT"
    adapter.model.forward = lambda wave: {"global_score": torch.tensor(.2)}
    with pytest.raises(AdapterError) as err:
        adapter.predict(audio(), "detect_and_localize", True)
    assert err.value.code == "INVALID_MODEL_OUTPUT"


def test_detection_only_never_claims_localization(config_path):
    modify(config_path, lambda c: (c.update(localization=None), c["thresholds"].update(localization=None)))
    adapter = TorchAdapter(config_path)
    assert adapter.describe()["capabilities"]["localization"] is False
    prediction = adapter.predict(audio(), "detect", False)
    assert prediction.result["timeline"] is None
    assert prediction.result["localization"]["status"] == "unsupported"
    assert prediction.result["localization"]["segments"] is None
    with pytest.raises(AdapterError) as err:
        adapter.predict(audio(), "detect_and_localize", True)
    assert err.value.code == "CAPABILITY_UNSUPPORTED"


def test_bct_layout_and_no_timeline_request(config_path):
    modify(config_path, lambda c: c["input"].update(layout="BCT", channels="preserve"))
    adapter = TorchAdapter(config_path)
    result = adapter.predict(audio(), "detect_and_localize", False).result
    assert result["timeline"] is None
    assert result["localization"]["status"] == "available"


def test_config_relative_paths_survive_moving_folder(config_path, tmp_path):
    import shutil
    destination = tmp_path / "moved"
    shutil.copytree(config_path.parent, destination)
    first = TorchAdapter(config_path)
    second = TorchAdapter(destination / "config.json")
    assert first.version == second.version
    assert second.predict(audio(), "detect", False).result["decision"] == first.predict(audio(), "detect", False).result["decision"]


def test_unknown_config_and_false_time_offset_are_rejected(config_path):
    original = config_path.read_text()
    modify(config_path, lambda c: c.update(stride=1600))
    with pytest.raises(ValueError, match="unknown keys"):
        TorchAdapter(config_path)
    config_path.write_text(original)
    modify(config_path, lambda c: c["localization"].update(start_offset_samples=False))
    with pytest.raises(ValueError, match="beginning at sample 0"):
        TorchAdapter(config_path)


def test_http_plugin_subprocess_checkpoint_to_audio_preview(config_path, tmp_path, monkeypatch):
    """HTTP lifecycle discovers the plugin and loads its checkpoint in real subprocesses."""
    import io
    import os
    from pathlib import Path
    import time
    import wave

    from fastapi.testclient import TestClient
    from voxguard.app import create_app
    from voxguard.settings import Settings

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("VOXGUARD_MODEL_CONFIG", str(config_path))
    monkeypatch.setenv("PYTHONPATH", str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    settings = Settings(data_dir=tmp_path / "service", enable_mock=False,
                        adapter_factory="model_integration.torch_adapter:create_adapter", task_timeout_sec=30)
    source = audio()
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as output:
        output.setnchannels(source.waveform.shape[0])
        output.setsampwidth(2)
        output.setframerate(source.sample_rate_hz)
        pcm = (np.clip(source.waveform.T, -1, 1 - 1 / 32768) * 32768).round().astype("<i2")
        output.writeframes(pcm.tobytes())
    with TestClient(create_app(settings)) as client:
        models = client.get("/api/v1/models")
        assert models.status_code == 200
        descriptors = models.json()["models"]
        assert len(descriptors) == 1
        descriptor = descriptors[0]
        assert descriptor["id"] == "example-plumbing-fixture"
        assert descriptor["is_mock"] is True
        assert "-cfg" in descriptor["version"] and "-w" in descriptor["version"]
        submitted = client.post("/api/v1/analyses", files={"file": ("signal.wav", encoded.getvalue(), "audio/wav")},
                                data={"model_id": descriptor["id"], "mode": "detect_and_localize", "include_timeline": "true"})
        assert submitted.status_code == 202, submitted.text
        task_id = submitted.json()["analysis_id"]
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            response = client.get(f"/api/v1/analyses/{task_id}")
            assert response.status_code == 200
            result = response.json()
            if result["status"] in ("succeeded", "failed"):
                break
            time.sleep(.05)
        assert result["status"] == "succeeded", result
        assert result["is_mock"] is True
        assert result["model"]["version"] == descriptor["version"]
        # Check the loaded parameters, not the module's uninitialized defaults;
        # PCM conversion introduces at most a small quantization difference.
        expected = ((.2 + .25 * .64) / .65) * 1.25 - .1
        assert result["result"]["decision"]["spoof_score"] == pytest.approx(expected, abs=1e-4)
        assert len(result["result"]["localization"]["segments"]) == 2
        assert result["result"]["timeline"][-1]["end_sec"] == pytest.approx(.65)
        preview = client.get(result["audio"]["preview_url"])
        assert preview.status_code == 200
        with wave.open(io.BytesIO(preview.content), "rb") as output:
            assert output.getframerate() == 16000
            assert output.getnframes() == source.waveform.shape[-1]
