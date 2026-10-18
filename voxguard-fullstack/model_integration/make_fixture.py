"""Generate a tiny real checkpoint for plumbing verification; no detector is trained."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def make_fixture(destination):
    import torch
    from .plumbing_fixture import PlumbingFixture

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / "source"
    source.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("plumbing_fixture.py"), source / "vg_plumbing_fixture_model.py")
    model = PlumbingFixture()
    with torch.no_grad():
        model.scale.fill_(1.25)
        model.bias.fill_(-0.1)
    weights = destination / "plumbing_fixture.pt"
    torch.save({"state_dict": model.state_dict()}, weights)
    config = {"schema_version": 1, "model_id": "example-plumbing-fixture", "display_name": "权重加载链路夹具（无检测能力）",
              "model_version": "example-plumbing-v1", "source_revision": "bundled-plumbing-v1", "is_mock": True,
              "source_dir": "source", "factory": "vg_plumbing_fixture_model:create_model", "model_kwargs": {"stride_samples": 1600},
              "weights": weights.name, "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "state_dict_key": "state_dict", "device": "cpu",
              "input": {"sample_rate_hz": 16000, "channels": "mean", "layout": "BT", "min_duration_sec": 0.5, "max_duration_sec": 120.0},
              "output": {"global_key": "global_score", "frame_key": "frame_scores", "score_direction": "higher_is_spoof"},
              "thresholds": {"detection": 0.25, "localization": 0.25, "version": "example-plumbing-threshold-v1"},
              "localization": {"stride_samples": 1600, "start_offset_samples": 0, "frame_layout": "BT",
                               "postprocessing_version": "example-threshold-adjacent-merge-v1"}}
    target = destination / "config.json"
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description="生成 is_mock=true 的权重加载链路夹具；不具备音频真假检测能力。")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(make_fixture(args.output))


if __name__ == "__main__":
    main()
