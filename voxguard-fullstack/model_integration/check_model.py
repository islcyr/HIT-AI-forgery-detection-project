"""Load configured weights and verify one audio prediction against the API validator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="校验模型加载、实际单音频推理及结果格式，不评估检测准确率。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--mode", choices=("detect", "detect_and_localize"), default="detect_and_localize")
    parser.add_argument("--output", help="可选：将实际输出保存为 JSON")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "backend"))
    from voxguard.audio import decode_audio
    from voxguard.settings import Settings
    from .torch_adapter import TorchAdapter

    adapter = TorchAdapter(args.config)
    source = Path(args.audio).expanduser().resolve()
    limits = Settings().limits()
    if source.stat().st_size > limits["max_upload_bytes"]:
        raise ValueError("音频文件超过服务上传限制。")
    with tempfile.TemporaryDirectory(prefix="voxguard-model-check-") as directory:
        audio = decode_audio(source, Path(directory), "model-check", adapter.describe(), limits)
        prediction = adapter.predict(audio, args.mode, include_timeline=args.mode == "detect_and_localize")
    # Use backend validation when running the check in the packaged application.
    from voxguard.schemas import Descriptor, validate_prediction
    descriptor = Descriptor.model_validate(adapter.describe())
    result, warnings = validate_prediction(prediction.result, descriptor, audio.duration_sec, args.mode)
    if type(prediction.analysis_sample_rate_hz) is not int or prediction.analysis_sample_rate_hz != descriptor.input.sample_rate_hz:
        raise ValueError("模型实际采样率与能力描述不一致。")
    output = {"model": adapter.describe(), "analysis_sample_rate_hz": prediction.analysis_sample_rate_hz, "result": result,
              "warnings": warnings,
              "notice": "本结果验证单音频推理和接口格式；不构成检测准确率验证。"}
    encoded = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
