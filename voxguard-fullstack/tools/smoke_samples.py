"""Submit bundled audio without labels; verify HTTP lifecycle and playable previews."""
import argparse
import io
import json
from pathlib import Path
import time
import wave

import httpx
from jsonschema import Draft202012Validator, FormatChecker
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-id", default="example-localizer")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    spec = yaml.safe_load((ROOT / "contracts/openapi.yaml").read_text())
    validator = Draft202012Validator({"$ref": "#/components/schemas/AnalysisEnvelope", "components": spec["components"]}, format_checker=FormatChecker())
    manifest = [json.loads(line) for line in (ROOT / "samples/manifest.jsonl").read_text().splitlines()]
    results = []
    with httpx.Client(base_url=args.base_url, timeout=30, trust_env=False) as client:
        catalog = client.get("/api/v1/models")
        catalog.raise_for_status()
        model = next(m for m in catalog.json()["models"] if m["id"] == args.model_id)
        assert model["status"] == "ready" and model["capabilities"]["localization"]
        for row in manifest[:args.limit]:
            path = ROOT / row["audio_path"]
            with path.open("rb") as audio:
                response = client.post("/api/v1/analyses", files={"file": (path.name, audio, "audio/wav")}, data={"model_id": args.model_id, "mode": "detect_and_localize", "include_timeline": "true"})
            assert response.status_code == 202, response.text
            record = response.json()
            deadline = time.monotonic() + 650
            while True:
                validator.validate(record)
                if record["status"] in ("succeeded", "failed"):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(record["analysis_id"])
                time.sleep(0.15)
                response = client.get(f'/api/v1/analyses/{record["analysis_id"]}')
                response.raise_for_status()
                record = response.json()
            assert record["status"] == "succeeded", record
            assert record["is_mock"] == model["is_mock"]
            assert record["model"]["version"] == model["version"]
            assert record["audio"]["duration_sec"] == row["duration_sec"]
            timeline = record["result"]["timeline"]
            assert timeline and timeline[0]["start_sec"] == 0
            assert timeline[-1]["end_sec"] == row["duration_sec"]
            preview = client.get(record["audio"]["preview_url"])
            preview.raise_for_status()
            with wave.open(io.BytesIO(preview.content)) as wav:
                assert wav.getnframes() / wav.getframerate() == row["duration_sec"]
            results.append({"sample_id": row["sample_id"], "analysis_id": record["analysis_id"], "is_mock": record["is_mock"], "status": record["status"]})
            print(f'{len(results)}/{min(args.limit, len(manifest))}: {row["sample_id"]} passed', flush=True)
    print(json.dumps({"validated": len(results), "model": model["id"], "is_mock": model["is_mock"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
