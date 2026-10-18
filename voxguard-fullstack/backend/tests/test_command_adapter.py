import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from model_integration.cfprf_output import convert_output, DECISION_RULE
from model_integration.command_adapter import CommandAdapter
from voxguard.adapters import AdapterError, AudioInput
from voxguard.schemas import Descriptor, validate_prediction

ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT / 'model_integration/cfprf-command.example.json').read_text())['models'][0]


def audio():
    return AudioInput(np.ones((1, 54400), dtype=np.float32), 16000, 3.4, 'fixture')


def raw():
    return {'model': 'CFPRF', 'checkpoint_variant': 'HAD', 'sample_rate': 16000,
            'duration': 3.4, 'is_fake': True, 'label': 'fake', 'score': 0.91,
            'score_type': 'max_refined_proposal_verification', 'decision_rule': DECISION_RULE,
            'frame_resolution': 0.02, 'segments': [{'start': 0.8, 'end': 1.6, 'score': 0.91}]}


def convert(data, mode='detect_and_localize'):
    c = config()
    prediction = convert_output(data, audio(), {**c['converter_options'], **c['descriptor']}, mode)
    return validate_prediction(prediction.result, Descriptor.model_validate(c['descriptor']), 3.4, mode)


def test_cfprf_preserves_proposals_without_timeline():
    result, warnings = convert(raw())
    assert not warnings
    assert result['timeline'] is None
    assert result['decision']['source_score_type'] == raw()['score_type']
    assert result['localization']['score_aggregation'] == 'proposal'
    assert result['localization']['segments'] == [{'start_sec': 0.8, 'end_sec': 1.6, 'spoof_score': 0.91}]


def test_strict_threshold_empty_real_and_detection_only():
    data = raw()
    data.update(is_fake=False, label='real', score=0.5, segments=[])
    for mode in ('detect', 'detect_and_localize'):
        result, warnings = convert(data, mode)
        assert result['decision']['label'] == 'bonafide'
        assert not warnings
    data.update(is_fake=True, label='fake')
    with pytest.raises(ValueError, match='Label and score'):
        convert(data)


@pytest.mark.parametrize('field,value', [('duration', 4.0), ('sample_rate', 8000), ('checkpoint_variant', 'other'), ('decision_rule', 'other'), ('frame_resolution', 0.01), ('label', 'real')])
def test_cfprf_rejects_mismatched_metadata(field, value):
    data = raw()
    data[field] = value
    with pytest.raises(ValueError):
        convert(data)


@pytest.mark.parametrize('segments', [
    [{'start': 0.8, 'end': 4, 'score': 0.91}],
    [{'start': 0.8, 'end': 1.6, 'score': float('nan')}],
    [{'start': 0.8, 'end': 1.6, 'score': 0.91}, {'start': 1.0, 'end': 2.0, 'score': 0.8}],
])
def test_proposal_validation_rejects_invalid_intervals(segments):
    data = raw()
    data['segments'] = segments
    with pytest.raises(ValueError):
        convert(data)


def test_command_uses_private_paths_and_no_shell(tmp_path, monkeypatch):
    c = config()
    c.update(cwd=str(tmp_path), required_files=[])
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        assert kwargs['cwd'] == tmp_path
        assert kwargs['check'] is True
        assert 'shell' not in kwargs
        assert Path(args[2]).read_bytes()[20:22] == b'\x03\x00'
        Path(args[-1]).write_text(json.dumps(raw()))
    monkeypatch.setattr(subprocess, 'run', run)
    adapter = CommandAdapter(c, tmp_path)
    for _ in range(2):
        assert adapter.predict(audio(), 'detect_and_localize', True).result['decision']['spoof_score'] == 0.91
    assert calls[0][2] != calls[1][2]
    assert not Path(calls[0][2]).exists()


@pytest.mark.parametrize('failure,code', [('timeout', 'MODEL_TIMEOUT'), ('exit', 'INFERENCE_FAILED'), ('json', 'INVALID_MODEL_OUTPUT')])
def test_command_errors_are_explicit(tmp_path, monkeypatch, failure, code):
    c = config()
    c.update(cwd=str(tmp_path), required_files=[])
    def run(args, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(args, 1)
        if failure == 'exit':
            raise subprocess.CalledProcessError(1, args)
        Path(args[-1]).write_text('not json')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(AdapterError) as error:
        CommandAdapter(c, tmp_path).predict(audio(), 'detect', False)
    assert error.value.code == code


def test_normalized_outputs_match_published_contract():
    from jsonschema import Draft202012Validator
    import yaml
    contract = yaml.safe_load((ROOT / 'contracts/openapi.yaml').read_text())
    for mode in ('detect', 'detect_and_localize'):
        result, _ = convert(raw(), mode)
        schema = {'$ref': '#/components/schemas/AnalysisResult', 'components': contract['components']}
        Draft202012Validator(schema).validate(result)
    schema = {'$ref': '#/components/schemas/Model', 'components': contract['components']}
    Draft202012Validator(schema).validate(Descriptor.model_validate(config()['descriptor']).model_dump())


def test_real_command_execution(tmp_path):
    import sys
    script = tmp_path / 'infer.py'
    script.write_text('import sys,json\nfrom pathlib import Path\nassert Path(sys.argv[1]).read_bytes()[:4] == b"RIFF"\nPath(sys.argv[2]).write_text(json.dumps(' + repr(raw()) + '))\n')
    c = config()
    c.update(cwd=str(tmp_path), required_files=[], command=[sys.executable, str(script), '{audio}', '{output}'])
    result = CommandAdapter(c, tmp_path).predict(audio(), 'detect_and_localize', True)
    assert result.result['localization']['segments'][0]['start_sec'] == 0.8


def test_factory_registers_multiple_independent_methods(tmp_path, monkeypatch):
    from model_integration.command_adapter import create_adapter
    configs = []
    for identifier in ('cfprf-one', 'other-method'):
        c = config()
        c.update(cwd=str(tmp_path), required_files=[])
        c['descriptor']['id'] = identifier
        configs.append(c)
    path = tmp_path / 'models.json'
    path.write_text(json.dumps({'models': configs}))
    monkeypatch.setenv('VOXGUARD_COMMAND_CONFIG', str(path))
    assert [a.describe()['id'] for a in create_adapter()] == ['cfprf-one', 'other-method']
