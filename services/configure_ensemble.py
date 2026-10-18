"""Generate a versioned five-model configuration from actual deployed artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = {r['path']: r for r in json.loads((ROOT/'temp/weight_manifest.json').read_text())}
    names = [f'CFPRF/{stage}_{variant}.pth' for variant in ['HAD','PS','LAVDF'] for stage in ['1FDN','2PRN']]
    names += ['BAM/model.ckpt', 'SAL/SAL_WavLM.ckpt', 'shared/xlsr2_300m.pt', 'shared/wavlm_large.pt']
    artifacts = [{'path': 'temp/weights/'+name, 'sha256': manifest[name]['sha256']} for name in names]
    code = ['services/model_router.py', 'services/ensemble_worker.py', 'services/serve_ensemble.py',
            'methods/cfprf/audio_forgery/cfprf.py', 'temp/CFPRF/models/FDN.py',
            'temp/CFPRF/models/PRN.py', 'temp/CFPRF/libs/tool.py',
            'voxguard-fullstack/model_integration/ensemble_output.py',
            'voxguard-fullstack/model_integration/ensemble_resident.py',
            'temp/weights/BAM/extracted/model.ckpt']
    for method, package in [('bam','bam_runtime'),('sal','sal_runtime')]:
        code.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'methods'/method/package).rglob('*.py')))
    for name in code:
        digest = hashlib.sha256()
        with (ROOT/name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        artifacts.append({'path': name, 'sha256': digest.hexdigest()})
    version = 'ensemble-five-v1-'+hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()[:16]
    descriptor = {'id': 'ensemble-five', 'version': version, 'display_name': '音频综合检测',
                  'status': 'ready', 'is_mock': False, 'decision_operator': 'gt',
                  'capabilities': {'detection': True, 'localization': True},
                  'input': {'sample_rate_hz': 16000, 'min_duration_sec': 0.5, 'max_duration_sec': 21.25},
                  'localization_resolution_sec': 0.16,
                  'thresholds': {'detection': 0.5, 'localization': 0.5, 'version': 'temporal-three-of-five-v1'}}
    path = ROOT/'voxguard-fullstack/model_integration/ensemble-resident.json'
    path.write_text(json.dumps({'descriptor': descriptor, 'artifacts': artifacts}, ensure_ascii=False, indent=2)+'\n')
    print(version)


if __name__ == '__main__':
    main()
