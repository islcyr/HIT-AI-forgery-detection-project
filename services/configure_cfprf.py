"""Bind the deployed model version to weights, inference code, and policy."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = {row['path']:row for row in json.loads((ROOT/'temp/weight_manifest.json').read_text())}
    artifacts = [{'path':'temp/weights/'+name,'sha256':manifest[name]['sha256']} for name in
                 ('CFPRF/1FDN_HAD.pth','CFPRF/2PRN_HAD.pth','shared/xlsr2_300m.pt')]
    for name in ('methods/cfprf/audio_forgery/cfprf.py','methods/cfprf/scripts/serve_cfprf.py',
                 'temp/CFPRF/models/FDN.py','temp/CFPRF/models/PRN.py','temp/CFPRF/libs/tool.py',
                 'voxguard-fullstack/model_integration/cfprf_output.py','voxguard-fullstack/model_integration/cfprf_resident.py'):
        artifacts.append({'path':name,'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()})
    identity = {'artifacts':artifacts,'checkpoint_variant':'HAD','torch':'2.5.1+cu121',
                'preprocessing':'ffmpeg-equal-mono-f32-16000-v1', 'threshold_operator':'gt','threshold':0.5,
                'max_duration_sec':21.25,'postprocessing':'cfprf-native-proposals-v1'}
    version = 'cfprf-had-v1-' + hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    descriptor = {
        'id':'cfprf-had','version':version,'display_name':'CFPRF (HAD)','status':'ready','is_mock':False,
        'decision_operator':'gt','capabilities':{'detection':True,'localization':True},
        'input':{'sample_rate_hz':16000,'min_duration_sec':0.5,'max_duration_sec':21.25},
        'localization_resolution_sec':0.02,
        'thresholds':{'detection':0.5,'localization':0.5,'version':'cfprf-verification-gt-0.5-v1'},
    }
    directory = ROOT/'voxguard-fullstack/model_integration'
    config = {**identity,'descriptor':descriptor}
    (directory/'cfprf-resident.json').write_text(json.dumps(config,indent=2)+'\n')
    command = json.loads((directory/'cfprf-command.example.json').read_text())
    entry = command['models'][0]
    entry.update(cwd='../..',descriptor=descriptor)
    entry['command'] = ['bash','methods/cfprf/scripts/cfprf.sh','{audio}','--variant','HAD','--device','cuda:0','--output','{output}']
    for name in ('cfprf-command.json','cfprf-command.example.json'):
        (directory/name).write_text(json.dumps(command,indent=2)+'\n')
    print(version)


if __name__ == '__main__':
    main()
