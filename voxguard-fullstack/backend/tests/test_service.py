from datetime import timedelta
import io
import json
from pathlib import Path
import subprocess
import time
import wave

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
import numpy as np
import pytest
import yaml

from voxguard.app import create_app
from voxguard.settings import Settings
from voxguard.tasks import iso, utc_now


def wav_bytes(duration=1.2, channels=1, silent=False, rate=16000):
    buffer = io.BytesIO()
    samples = np.zeros(round(duration * rate)) if silent else 10000 * np.sin(np.arange(round(duration * rate)) * 2 * np.pi * 440 / rate)
    pcm = np.repeat(samples.astype("<i2")[:, None], channels, axis=1)
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as test_client:
        yield test_client


def submit(client, audio=None, **fields):
    return client.post("/api/v1/analyses", data={"model_id": "example-localizer", **fields},
                       files={"file": ("input.wav", wav_bytes() if audio is None else audio, "audio/wav")})


def finish(client, response):
    assert response.status_code == 202, response.text
    record = response.json()
    deadline = time.monotonic() + 20
    while record["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.03)
        response = client.get("/api/v1/analyses/" + record["analysis_id"])
        assert response.status_code == 200
        record = response.json()
    assert record["status"] in {"succeeded", "failed"}, record
    return record


def assert_contract(value, schema_name):
    contract_path = Path(__file__).resolve().parents[2] / "contracts/openapi.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    Draft202012Validator({"$ref": f"#/components/schemas/{schema_name}", "components": contract["components"]}).validate(value)


def test_mock_lifecycle_and_real_preview(client):
    catalog = client.get("/api/v1/models").json()
    assert_contract(catalog, "ModelsResponse")
    assert len(catalog["models"]) == 2 and all(item["is_mock"] for item in catalog["models"])
    response = submit(client, include_timeline="true")
    assert_contract(response.json(), "AnalysisEnvelope")
    record = finish(client, response)
    assert record["status"] == "succeeded", record
    assert record["is_mock"] and record["result"]["decision"]["spoof_score"] == 1.6
    assert_contract(record, "AnalysisEnvelope")
    assert record["result"]["timeline"][0]["start_sec"] == 0
    assert record["result"]["timeline"][-1]["end_sec"] == 1.2
    assert record["audio"]["duration_sec"] == 1.2
    audio = client.get(record["audio"]["preview_url"])
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(audio.content)) as decoded:
        assert decoded.getnframes() / decoded.getframerate() == 1.2
    assert client.get("/api/v1/analyses/" + record["analysis_id"]).json() == record


@pytest.mark.parametrize("model_id,expected", [("example-localizer", "not_requested"), ("example-detector", "unsupported")])
def test_detection_only_states(client, model_id, expected):
    record = finish(client, submit(client, model_id=model_id, mode="detect"))
    assert record["status"] == "succeeded"
    assert record["result"]["localization"]["status"] == expected
    assert record["result"]["localization"]["segments"] is None
    assert record["result"]["timeline"] is None
    assert_contract(record, "AnalysisEnvelope")


def test_localization_timeline_transport_optional(client):
    record = finish(client, submit(client))
    assert record["status"] == "succeeded"
    assert record["result"]["localization"]["segments"]
    assert record["result"]["timeline"] is None


@pytest.mark.parametrize("audio,error", [(b"broken file", "INVALID_AUDIO"), (wav_bytes(silent=True), "NO_VALID_AUDIO"),
                                          (wav_bytes(duration=0.2), "DURATION_OUT_OF_RANGE"), (wav_bytes(channels=3), "CHANNELS_UNSUPPORTED")],
                         ids=["corrupt", "silent", "short", "channels"])
def test_decoding_failures_are_failed_tasks(client, audio, error):
    record = finish(client, submit(client, audio))
    assert record["status"] == "failed" and record["result"] is None
    assert record["error"]["code"] == error
    assert_contract(record, "AnalysisEnvelope")
    assert client.get(f'/api/v1/analyses/{record["analysis_id"]}/audio').status_code == 409


def test_request_errors_are_uniform(client):
    cases = [(submit(client, b""), 422, "NO_VALID_AUDIO"),
             (submit(client, model_id="unknown"), 404, "MODEL_NOT_FOUND"),
             (submit(client, model_id="example-detector"), 422, "CAPABILITY_UNSUPPORTED"),
             (submit(client, mode="detect", include_timeline="true"), 422, "INVALID_REQUEST"),
             (submit(client, include_timeline="yes"), 422, "INVALID_REQUEST"),
             (client.post("/api/v1/analyses", json={}), 400, "INVALID_REQUEST"),
             (client.get("/api/v1/analyses/missing"), 404, "ANALYSIS_NOT_FOUND")]
    for response, status, code in cases:
        assert response.status_code == status, response.text
        assert response.json()["error"]["code"] == code
        assert_contract(response.json(), "ErrorResponse")


def test_upload_limit(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_upload_bytes=1024))) as client:
        response = submit(client, b"x" * 1025)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "FILE_TOO_LARGE"
        assert not list(tmp_path.glob("upload-*.tmp"))


def test_format_comes_from_content(client, tmp_path):
    source, encoded = tmp_path / "source.wav", tmp_path / "actual.ogg"
    source.write_bytes(wav_bytes())
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), str(encoded)], check=True)
    record = finish(client, submit(client, encoded.read_bytes()))
    assert record["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_original_rate_and_analysis_rate_are_distinct(client):
    record = finish(client, submit(client, wav_bytes(rate=44100)))
    assert record["status"] == "succeeded"
    assert record["audio"]["original_sample_rate_hz"] == 44100
    assert record["audio"]["analysis_sample_rate_hz"] == 16000
    assert record["audio"]["duration_sec"] == 1.2


def test_expiry_removes_audio_but_keeps_tombstone(client):
    record = finish(client, submit(client))
    task_id = record["analysis_id"]
    manager = client.app.state.manager
    manager.records[task_id]["expires_at"] = iso(utc_now() - timedelta(seconds=1))
    response = client.get("/api/v1/analyses/" + task_id)
    assert response.status_code == 410
    directory = manager.settings.data_dir / task_id
    assert [x.name for x in directory.iterdir()] == ["record.json"]
    assert client.get(record["audio"]["preview_url"]).status_code == 410


def test_terminal_result_persists_across_restart(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        record = finish(client, submit(client))
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/analyses/" + record["analysis_id"]).json() == record
        assert client.get(record["audio"]["preview_url"]).status_code == 200


def test_no_real_model_means_empty_catalog(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, enable_mock=False))) as client:
        assert client.get("/api/v1/models").json()["models"] == []
        assert submit(client).status_code == 404


def test_invalid_adapter_output_fails(tmp_path):
    settings = Settings(data_dir=tmp_path, enable_mock=False, adapter_factory="tests.fake_adapters:broken_factory")
    with TestClient(create_app(settings)) as client:
        record = finish(client, submit(client))
        assert record["error"]["code"] == "INVALID_MODEL_OUTPUT"


def test_hung_inference_is_killed_and_queue_is_bounded(tmp_path):
    settings = Settings(data_dir=tmp_path, enable_mock=False, adapter_factory="tests.fake_adapters:slow_factory",
                        queue_capacity=1, task_timeout_sec=1.5, retention_sec=60)
    with TestClient(create_app(settings)) as client:
        first = submit(client)
        second = submit(client)
        assert second.status_code == 429 and second.headers["retry-after"] == "3"
        record = finish(client, first)
        assert record["error"]["code"] == "INFERENCE_TIMEOUT"
        assert client.app.state.manager.pending == 0


def test_static_files_are_whitelisted(client):
    for path in ("/", "/app.js", "/analysis.js", "/api.js", "/styles.css", "/openapi.json"):
        assert client.get(path).status_code == 200
    for path in ("/samples/manifest.jsonl", "/data", "/backend/requirements.txt"):
        assert client.get(path).status_code == 404
