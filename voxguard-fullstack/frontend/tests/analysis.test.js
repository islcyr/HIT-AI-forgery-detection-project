import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DEFAULT_LIMITS, normalizeReport, normalizeModels, validateFile, validateDuration, effectiveLimits, apiURL, previewURL, formatScore, timelineLabels } from '../analysis.js';

test('short audio timeline labels are distinct and include fractional seconds', () => {
  for (const duration of [0.5, 1.1, 1.97625, 2, 4.9, 5, 21.25, 60]) {
    const labels = timelineLabels(duration);
    assert.equal(new Set(labels).size, labels.length);
    assert.ok(labels.length >= 2 && labels.length <= 6);
    if (duration < 5) assert.ok(labels.every((label) => label.includes('.')));
  }
  assert.deepEqual(timelineLabels(2), ['00:00.0', '00:00.4', '00:00.8', '00:01.2', '00:01.6', '00:02.0']);
  assert.deepEqual(timelineLabels(0), []);
});

const fixture = (name) => JSON.parse(readFileSync(new URL(`../../examples/${name}.json`, import.meta.url)));

test('all contract success examples parse without converting or merging scores', () => {
  for (const name of ['analysis-bonafide', 'analysis-detection-only', 'analysis-localized']) {
    const envelope = fixture(name), report = normalizeReport(envelope);
    assert.equal(report.score, envelope.result.decision.spoof_score);
    assert.equal(report.threshold, envelope.result.decision.threshold);
    assert.equal(report.envelope, envelope);
    assert.equal(report.segments.length, envelope.result.localization.segments?.length ?? 0);
    for (const [i, segment] of report.segments.entries()) {
      assert.deepEqual(segment, { start: envelope.result.localization.segments[i].start_sec, end: envelope.result.localization.segments[i].end_sec, score: envelope.result.localization.segments[i].spoof_score });
    }
  }
});

test('negative and large scores remain raw, with no percent formatting', () => {
  for (const score of [-123.45, 1.6, 1001]) {
    const e = fixture('analysis-localized'); e.result.decision.spoof_score = score;
    assert.equal(normalizeReport(e).score, score);
    assert.equal(formatScore(score).includes('%'), false);
  }
});

test('unavailable localization remains distinct from an empty prediction', () => {
  const empty = fixture('analysis-bonafide');
  assert.equal(normalizeReport(empty).localizationStatus, 'available');
  assert.equal(normalizeReport(empty).suspiciousDuration, 0);
  const unsupported = fixture('analysis-detection-only');
  assert.equal(normalizeReport(unsupported).localizationStatus, 'unsupported');
  assert.equal(normalizeReport(unsupported).suspiciousDuration, null);
  unsupported.result.localization.status = 'not_requested';
  assert.equal(normalizeReport(unsupported).localizationStatus, 'not_requested');
});

test('rejects ambiguous provenance, nonfinite scores, overlap, and out of range segments', () => {
  for (const mutate of [
    (e) => delete e.is_mock,
    (e) => { e.result.decision.spoof_score = Infinity; },
    (e) => { e.result.localization.segments[1].start_sec = e.result.localization.segments[0].start_sec; },
    (e) => { e.result.localization.segments[0].end_sec = e.audio.duration_sec + 1; },
    (e) => { e.status = 'running'; e.result = null; }
  ]) {
    const envelope = fixture('analysis-localized'); mutate(envelope); assert.throws(() => normalizeReport(envelope));
  }
});

test('service and model constraints intersect; file and duration failures are detected', () => {
  const limits = effectiveLimits(DEFAULT_LIMITS, { input: { min_duration_sec: 2, max_duration_sec: 30 } });
  assert.equal(limits.min_duration_sec, 2); assert.equal(limits.max_duration_sec, 30);
  assert.throws(() => validateDuration(1, limits)); assert.throws(() => validateDuration(31, limits));
  assert.throws(() => validateFile({ name: 'x.ogg', size: 100 }));
  assert.throws(() => validateFile({ name: 'x.wav', size: 0 }));
  assert.throws(() => validateFile({ name: 'x.wav', size: 20971521 }));
  assert.doesNotThrow(() => validateFile({ name: 'x.WAV', size: 20971520 }));
});

test('model catalogs require explicit mock provenance and allow an empty catalog', () => {
  assert.equal(normalizeModels({ limits: DEFAULT_LIMITS, models: [] }).models.length, 0);
  const model = { id: 'mock', version: 'v1', display_name: '模拟', status: 'ready', is_mock: true, capabilities: { detection: true, localization: false }, input: { min_duration_sec: .5, max_duration_sec: 120 } };
  assert.equal(normalizeModels({ limits: DEFAULT_LIMITS, models: [model] }).models[0].is_mock, true);
  delete model.is_mock;
  assert.throws(() => normalizeModels({ limits: DEFAULT_LIMITS, models: [model] }));
});

test('API and preview paths resolve against the configured backend origin', () => {
  assert.equal(apiURL('/api/v1', 'models', 'http://localhost:8000/'), 'http://localhost:8000/api/v1/models');
  assert.equal(previewURL('http://server:8000/api/v1', '/api/v1/analyses/123/audio', 'http://localhost:5173'), 'http://server:8000/api/v1/analyses/123/audio');
  assert.equal(previewURL('http://server:8000/api/v1', 'analyses/123/audio'), 'http://server:8000/api/v1/analyses/123/audio');
  assert.throws(() => previewURL('http://server:8000/api/v1', 'http://another/audio'));
});

test('CFPRF proposal results preserve strict threshold and native score semantics', () => {
  const e = fixture('analysis-localized');
  e.result.timeline = null;
  e.result.decision.threshold_operator = 'gt';
  e.result.decision.source_score_type = 'max_refined_proposal_verification';
  e.result.localization.score_aggregation = 'proposal';
  const report = normalizeReport(e);
  assert.equal(report.thresholdOperator, 'gt');
  assert.equal(report.sourceScoreType, 'max_refined_proposal_verification');
  assert.equal(report.scoreAggregation, 'proposal');
  assert.equal(report.segments.length, e.result.localization.segments.length);
  e.result.decision.threshold_operator = 'unknown';
  assert.throws(() => normalizeReport(e));
});
