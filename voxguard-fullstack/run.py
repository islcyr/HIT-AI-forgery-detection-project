"""Start the single-process API and frontend from the unpacked project."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def load_env(path):
    if not path.is_file():
        return
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip().replace("_", "a").isalnum():
            raise ValueError(f"Invalid configuration at {path.name}:{number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def main():
    parser = argparse.ArgumentParser(description="VoxGuard 音频检测与定位服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    os.chdir(ROOT)
    load_env(args.env_file.resolve())
    sys.path.insert(0, str(ROOT / "backend"))
    import uvicorn
    uvicorn.run("voxguard.app:create_app", factory=True, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
