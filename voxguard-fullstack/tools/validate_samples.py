"""Verify portable sample metadata, checksums, time labels and API examples."""
from pathlib import Path
import csv
import collections
import hashlib
import json
import math
import wave

from jsonschema import Draft202012Validator, FormatChecker
import yaml

root = Path(__file__).resolve().parents[1]
samples = root / 'samples'

def local(path):
    assert path.startswith('samples/')
    return samples / path[len('samples/'):]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest_schema = json.loads((root / 'contracts/sample-manifest.schema.json').read_text())
detail_schema = json.loads((root / 'contracts/detailed-label.schema.json').read_text())
for schema in [manifest_schema, detail_schema]:
    Draft202012Validator.check_schema(schema)
manifest_validator = Draft202012Validator(manifest_schema, format_checker=FormatChecker())
detail_validator = Draft202012Validator(detail_schema, format_checker=FormatChecker())
records = [json.loads(line) for line in (samples / 'manifest.jsonl').read_text().splitlines()]
assert len(records) == 30
summary_csv = list(csv.DictReader((samples / 'labels/summary.csv').open(encoding='utf-8-sig')))
segments_csv = list(csv.DictReader((samples / 'labels/segments.csv').open(encoding='utf-8-sig')))
assert len(summary_csv) == 30
assert len(list((samples / 'audio').glob('*.wav'))) == 30
assert len(list((samples / 'labels/detailed').glob('*.json'))) == 30
assert len(list((samples / 'labels/original').iterdir())) == 30
csv_summary = {row['sample_id']: row for row in summary_csv}
csv_segments = collections.defaultdict(list)
for row in segments_csv:
    csv_segments[row['sample_id']].append(row)
categories = collections.Counter()
hashes = set()
normalization_count = 0
unknown_seconds = []
total_duration = []
for record in records:
    manifest_validator.validate(record)
    assert record['is_mock'] is False
    detail = json.loads(local(record['detailed_label_path']).read_text())
    detail_validator.validate(detail)
    assert record['sample_id'] == detail['sample_id']
    categories[detail['category']] += 1
    audio = local(record['audio_path'])
    digest = sha(audio)
    hashes.add(digest)
    assert digest == record['audio_sha256'] == record['source']['original_audio_sha256'] == detail['audio']['sha256']
    assert sha(local(record['ground_truth']['annotation_source'])) == record['ground_truth']['annotation_sha256'] == detail['annotation']['source_sha256']
    with wave.open(str(audio), 'rb') as w:
        assert w.getframerate() == record['sample_rate_hz'] == detail['audio']['sample_rate_hz'] == 16000
        assert w.getnchannels() == record['channels'] == detail['audio']['channels'] == 1
        assert w.getsampwidth() * 8 == detail['audio']['bits_per_sample'] == 16
        assert w.getnframes() == detail['audio']['num_samples']
        duration = w.getnframes() / w.getframerate()
        assert duration == record['duration_sec'] == detail['audio']['duration_sec']
        pcm = w.readframes(w.getnframes())
        assert len(pcm) == w.getnframes() * w.getsampwidth()
        assert any(pcm), 'All-zero audio is not an annotated speech example.'
    total_duration.append(duration)
    last = 0.0
    for index, segment in enumerate(detail['segments']):
        assert segment['index'] == index
        assert segment['start_sec'] == last < segment['end_sec'] <= duration
        last = segment['end_sec']
        if segment['label'] == 'unannotated':
            assert segment['source_label'] is None and segment['source_start_sec'] is None and segment['source_end_sec'] is None
    assert last == duration
    for label, field in [('spoof', 'spoof_segments'), ('bonafide', 'bonafide_segments'), ('unannotated', 'unannotated_segments'), ('non_speech', 'non_speech_segments')]:
        group = [s for s in detail['segments'] if s['label'] == label]
        assert detail['summary'][label + '_segment_count'] == len(group)
        assert abs(detail['summary'][label + '_duration_sec'] - math.fsum(s['end_sec'] - s['start_sec'] for s in group)) < 1e-12
        if label == 'non_speech' and not detail['annotation']['non_speech_annotation_available']:
            assert record['ground_truth'][field] is None
        else:
            assert record['ground_truth'][field] == [{'start_sec': s['start_sec'], 'end_sec': s['end_sec']} for s in group]
    unknown_seconds.append(detail['summary']['unannotated_duration_sec'])
    category = detail['category']
    if category == 'bonafide':
        assert detail['utterance_label'] == 'bonafide' and detail['summary']['spoof_segment_count'] == 0
    elif category == 'partialspoof':
        assert detail['summary']['spoof_segment_count'] > 0 and detail['summary']['bonafide_segment_count'] > 0
    else:
        assert detail['source']['dataset'] == 'LlamaPartialSpoof'
        assert detail['source']['original_split'] == 'not_defined' and detail['source']['speaker_id'] is None
        assert not detail['annotation']['non_speech_annotation_available']
        assert detail['summary']['spoof_ratio_annotated_speech'] is None
        assert len(detail['segments']) == 1 and detail['segments'][0]['label'] == 'spoof'
        original = local(detail['annotation']['source_file']).read_text().split()
        assert original[0] == record['sample_id'] and len(original) == 4
        start, end, label = original[3].split('-')
        assert float(start) == 0 and label == 'spoof'
        assert abs(float(end) - duration) <= 0.00005 + 1e-12
        assert detail['segments'][0]['source_end_sec'] == float(end)
        for change in detail['annotation']['normalizations']:
            normalization_count += 1
            assert change['original_end_sec'] == float(end) and change['normalized_end_sec'] == duration
            assert abs(change['delta_sec']) <= 0.00005 + 1e-12
    row = csv_summary[record['sample_id']]
    for key in ['category', 'utterance_label']:
        assert row[key] == detail[key]
    for key, value in detail['summary'].items():
        assert row[key] == ('' if value is None else str(value))
    assert len(csv_segments[record['sample_id']]) == len(detail['segments'])
    for row, segment in zip(csv_segments[record['sample_id']], detail['segments']):
        for key, value in segment.items():
            assert row[key] == ('' if value is None else str(value))
assert categories == {'bonafide': 10, 'fullspoof': 10, 'partialspoof': 10}
assert len(hashes) == 30
for record in [json.loads(line) for line in (root / 'examples/sample-manifest.jsonl').read_text().splitlines()]:
    manifest_validator.validate(record)
spec = yaml.safe_load((root / 'contracts/openapi.yaml').read_text())
response_schema = {'$ref': '#/components/schemas/AnalysisEnvelope', 'components': spec['components']}
api_validator = Draft202012Validator(response_schema, format_checker=FormatChecker())
for path in (root / 'examples').glob('analysis-*.json'):
    api_validator.validate(json.loads(path.read_text()))
print(json.dumps({'validated_audio': len(records), 'unique_audio_hashes': len(hashes), 'category_counts': dict(categories), 'total_duration_sec': math.fsum(total_duration), 'segments': len(segments_csv), 'max_unannotated_duration_sec': max(unknown_seconds), 'rounded_end_adjustments': normalization_count, 'checks': ['Draft202012 schemas', 'bundled WAV/annotation hashes and format', 'interval coverage and ordering', 'manifest/detailed JSON/CSV consistency', 'mock response schemas']}, ensure_ascii=False, indent=2))
