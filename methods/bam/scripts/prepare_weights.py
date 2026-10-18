"""Unpack the official download, whose .ckpt suffix actually contains a zip."""
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
archive = ROOT / 'temp/weights/BAM/model.ckpt'
expected = '5c7379d23028606d3beeb8b1aa5c340395fcc403ce70a2ca559ea322ab2994d1'
digest = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != expected:
    raise SystemExit('BAM archive checksum mismatch.')
target = archive.parent / 'extracted'
target.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as bundle:
    for name in ('model.ckpt', 'hparams.yaml'):
        # Extract only these literal paths; never extract an arbitrary tree.
        with bundle.open(name) as source, (target / name).open('wb') as output:
            import shutil
            shutil.copyfileobj(source, output)
print(f'BAM checkpoint ready: {target / "model.ckpt"}')
