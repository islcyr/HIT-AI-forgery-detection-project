"""One disposable, killable process per task. Invoke through the task manager."""
import json
import logging
from pathlib import Path
import sys

from .adapters import AdapterError, Prediction, load_adapters
from .audio import decode_audio
from .schemas import validate_prediction


def execute(task_dir: Path):
    config = json.loads((task_dir / "request.json").read_text(encoding="utf-8"))
    try:
        registry = load_adapters(config["enable_mock"], config["adapter_factory"])
        entry = registry.get(config["descriptor"]["id"])
        if entry is None or entry[1].model_dump() != config["descriptor"]:
            raise AdapterError("MODEL_UNAVAILABLE", "任务绑定的模型版本当前不可用。", True)
        adapter, descriptor = entry
        audio = decode_audio(task_dir / "upload.bin", task_dir, config["analysis_id"], config["descriptor"], config["limits"])
        # Localization always exposes internal bins to the service for validation;
        # include_timeline controls only whether they are sent over HTTP.
        prediction = adapter.predict(audio, config["mode"], config["mode"] == "detect_and_localize")
        try:
            if not isinstance(prediction, Prediction) or type(prediction.analysis_sample_rate_hz) is not int:
                raise ValueError("Adapter must return Prediction with an integer actual sample rate")
            if prediction.analysis_sample_rate_hz != descriptor.input.sample_rate_hz:
                raise ValueError("Actual analysis sample rate differs from descriptor")
            result, warnings = validate_prediction(prediction.result, descriptor, audio.duration_sec, config["mode"])
        except Exception as exc:
            logging.exception("Invalid model output for %s", config["analysis_id"])
            raise AdapterError("INVALID_MODEL_OUTPUT", "模型输出不符合接口约定。") from exc
        if not config["include_timeline"]:
            result["timeline"] = None
        payload = {"result": result, "audio": {"duration_sec": audio.duration_sec,
                   "original_sample_rate_hz": audio.sample_rate_hz, "original_channels": audio.waveform.shape[0],
                   "analysis_sample_rate_hz": prediction.analysis_sample_rate_hz,
                   "preview_url": f'/api/v1/analyses/{config["analysis_id"]}/audio'}, "warnings": warnings}
    except AdapterError as exc:
        payload = {"error": exc.detail()}
    except Exception:
        logging.exception("Inference failed for %s", config["analysis_id"])
        payload = {"error": {"code": "INFERENCE_FAILED", "message": "模型推理失败，请查看服务日志。", "retryable": False}}
    temp = task_dir / "output.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(task_dir / "output.json")


if __name__ == "__main__":
    if sys.argv[1] == "--describe":
        config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        registry = load_adapters(config["enable_mock"], config["adapter_factory"])
        Path(sys.argv[3]).write_text(json.dumps([entry[1].model_dump() for entry in registry.values()], ensure_ascii=False, allow_nan=False), encoding="utf-8")
    else:
        execute(Path(sys.argv[1]))
