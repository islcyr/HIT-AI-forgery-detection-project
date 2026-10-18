"""Verify bundled SHA-256 manifest using Python's standard library."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
count = 0
for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    expected, name = line.split("  ", 1)
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SystemExit(f"Missing or invalid path: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Checksum mismatch: {name}")
    count += 1
print(f"Verified {count} bundled files.")
