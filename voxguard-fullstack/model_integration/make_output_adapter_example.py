"""Generate portable source, wrapper checkpoint and config for a MOCK example."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def make_output_adapter_example(destination):
    import torch
    from .output_adapter_example import create_model

    destination = Path(destination).resolve()
    source_path = destination / "source" / "vg_output_adapter_example.py"
    weights = destination / "output_adapter_example.pt"
    config_path = destination / "config.json"
    for path in (source_path, weights, config_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing example file: {path}")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = Path(__file__).with_name("output_adapter_example.py").read_bytes()
    source_path.write_bytes(source_bytes)
    model = create_model(stride_samples=1600).cpu()
    # Save the wrapper's full state_dict, including the network.* key prefix.
    # TorchAdapter subsequently verifies this with strict=True.
    torch.save({"state_dict": model.state_dict()}, weights)
    config = {
        "schema_version": 1,
        "model_id": "example-output-adapter",
        "display_name": "输出适配示例（模拟联调，无检测能力）",
        "model_version": "example-output-adapter-v1",
        "source_revision": "bundled-output-adapter-v1-sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        "is_mock": True,
        "source_dir": "source",
        "factory": "vg_output_adapter_example:create_model",
        "model_kwargs": {"stride_samples": 1600},
        "weights": weights.name,
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "state_dict_key": "state_dict",
        "device": "cpu",
        "input": {
            "sample_rate_hz": 16000, "channels": "mean", "layout": "BT",
            "min_duration_sec": 0.5, "max_duration_sec": 120.0,
        },
        "output": {"global_key": "global_score", "frame_key": "frame_scores", "score_direction": "higher_is_spoof"},
        "thresholds": {"detection": 0.5, "localization": 0.5, "version": "example-output-adapter-threshold-v1"},
        "localization": {
            "stride_samples": 1600, "start_offset_samples": 0, "frame_layout": "BT",
            "postprocessing_version": "example-output-adapter-adjacent-merge-v1",
        },
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="生成 is_mock=true 的 tuple/logits 输出适配示例；没有音频伪造检测能力。")
    parser.add_argument("--output", required=True, help="生成目录；不会覆盖其中已有的示例文件")
    args = parser.parse_args()
    print(make_output_adapter_example(args.output))


if __name__ == "__main__":
    main()
