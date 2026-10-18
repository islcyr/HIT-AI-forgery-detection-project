import json
from pathlib import Path
import sys

import numpy as np
import pytest
from voxguard.adapters import AudioInput
from voxguard.schemas import Descriptor, validate_prediction
from model_integration.ensemble_output import convert_output
from model_integration.ensemble_resident import EnsembleAdapter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from services.model_router import MODEL_SPECS, fuse_results


def fixture():
    source = {}
    for model_id, _, name, variant in MODEL_SPECS:
        positive = model_id != 'cfprf-lavdf'
        source[model_id] = {'model': name, 'checkpoint_variant': variant,
                            'sample_rate': 16000, 'duration': 49666/16000,
                            'label': 'fake' if positive else 'real', 'is_fake': positive,
                            'score': 0.9 if positive else 0.0,
                            'segments': [{'start': 0.4, 'end': 2.2, 'score': 0.9}] if positive else []}
    raw = fuse_results(source)
    config = json.loads((ROOT/'voxguard-fullstack/model_integration/ensemble-resident.json').read_text())
    audio = AudioInput(np.zeros((1, 49666), dtype=np.float32), 16000, 49666/16000, 'test')
    return raw, config, audio


def test_ensemble_converts_vote_segments_without_frontend_contract_changes():
    raw, config, audio = fixture()
    prediction = convert_output(raw, audio, config['descriptor'], 'detect_and_localize')
    validate_prediction(prediction.result, Descriptor.model_validate(config['descriptor']), audio.duration_sec, 'detect_and_localize')
    assert prediction.result['decision']['spoof_score'] == raw['score']
    assert prediction.result['localization']['segments'][0]['start_sec'] == raw['segments'][0]['start']
    assert prediction.result['timeline'][-1]['end_sec'] == audio.duration_sec
    detect = convert_output(raw, audio, config['descriptor'], 'detect')
    validate_prediction(detect.result, Descriptor.model_validate(config['descriptor']), audio.duration_sec, 'detect')
    assert detect.result['timeline'] is None


def test_all_real_and_small_resampling_duration_difference():
    raw, config, audio = fixture()
    for result in raw['model_results'].values():
        result.update(segments=[], label='real', is_fake=False, score=0.0)
    raw = fuse_results(raw['model_results'])
    audio = AudioInput(audio.waveform, 16000, audio.duration_sec + 0.00001, 'test')
    prediction = convert_output(raw, audio, config['descriptor'], 'detect_and_localize')
    validate_prediction(prediction.result, Descriptor.model_validate(config['descriptor']), audio.duration_sec, 'detect_and_localize')
    assert prediction.result['decision']['label'] == 'bonafide'
    assert prediction.result['localization']['segments'] == []


def test_tampered_fusion_rejected():
    raw, config, audio = fixture()
    raw['score'] = 1.0
    with pytest.raises(ValueError):
        convert_output(raw, audio, config['descriptor'], 'detect_and_localize')


def test_adapter_wav_protocol_and_version_binding(monkeypatch):
    raw, config, audio = fixture()
    def request(self, path, body=None, timeout=540):
        if path == '/health':
            return {'ready': True, 'descriptor': config['descriptor']}
        assert body[:4] == b'RIFF'
        return {'result': raw}
    monkeypatch.setattr(EnsembleAdapter, '_request', request)
    adapter = EnsembleAdapter(config, 'http://127.0.0.1:8765')
    assert adapter.describe()['id'] == 'ensemble-five'
    assert adapter.predict(audio, 'detect_and_localize', False).result['decision']['label'] == 'spoof'
