import json
from pathlib import Path
import struct

import numpy as np
import pytest
from voxguard.adapters import AudioInput
from model_integration.cfprf_resident import ResidentAdapter

ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT/'model_integration/cfprf-resident.json').read_text())


def test_resident_protocol_and_model_version(monkeypatch):
    cfg = config()
    calls = []
    def request(self,path,body=None,timeout=540):
        calls.append(path)
        if path == '/health':
            return {'ready':True,'descriptor':cfg['descriptor'],'model_load_count':1,'pid':123}
        assert body[:4] == b'RIFF'
        assert struct.unpack('<H',body[20:22])[0] == 3  # IEEE float PCM
        samples = np.frombuffer(body[44:],dtype='<f4')
        np.testing.assert_allclose(samples,0.5)
        return {'result':{'model':'CFPRF','checkpoint_variant':'HAD','sample_rate':16000,
                         'duration':1.0,'is_fake':False,'label':'real','score':0.0,
                         'score_type':'max_refined_proposal_verification',
                         'decision_rule':'at_least_one_refined_proposal_with_verification_gt_0.5',
                         'frame_resolution':0.02,'segments':[]}}
    monkeypatch.setattr(ResidentAdapter,'_request',request)
    adapter = ResidentAdapter(cfg,'http://127.0.0.1:8765')
    audio = AudioInput(np.array([[0.25]*16000,[0.75]*16000],dtype=np.float32),16000,1.0,'id')
    for _ in range(2):
        output = adapter.predict(audio,'detect_and_localize',False)
        assert output.result['decision']['label'] == 'bonafide'
    assert calls == ['/health','/predict','/predict']
    assert adapter.describe()['input']['max_duration_sec'] == 21.25
    assert not adapter.describe()['is_mock']


def test_resident_refuses_wrong_model_version(monkeypatch):
    cfg = config()
    wrong = {**cfg['descriptor'],'version':'wrong'}
    monkeypatch.setattr(ResidentAdapter,'_request',lambda *a,**k:{'ready':True,'descriptor':wrong})
    with pytest.raises(ValueError,match='version'):
        ResidentAdapter(cfg,'http://127.0.0.1:8765')
