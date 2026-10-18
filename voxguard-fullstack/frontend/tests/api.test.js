import test from 'node:test';
import assert from 'node:assert/strict';
import { request, pollAnalysis, retryDelay, wait } from '../api.js';

const response = (payload, status = 200, headers = {}) => new Response(JSON.stringify(payload), { status, headers });
const queued = () => ({ analysis_id: 'task-1', status: 'queued', mode: 'detect', model: { id: 'mock', version: 'v1' }, is_mock: true, created_at: '2026-10-13T00:00:00Z', completed_at: null, expires_at: '2026-10-14T00:00:00Z', warnings: [], result: null, error: null });

test('GET honors Retry-After and has a bounded retry count', async () => {
  let calls = 0; const delays = [];
  const result = await request('http://example/models', { fetchImpl: async () => ++calls < 3 ? response({ error: { message: 'busy' } }, 503, { 'Retry-After': '2' }) : response({ models: [] }), sleep: async (ms) => delays.push(ms) });
  assert.deepEqual(result, { models: [] }); assert.equal(calls, 3); assert.deepEqual(delays, [2000, 2000]);
  calls = 0;
  await assert.rejects(request('http://example/models', { fetchImpl: async () => { calls++; return response({ error: { message: 'busy' } }, 429); }, sleep: async () => {} }), /busy/);
  assert.equal(calls, 4);
});

test('POST is never automatically repeated', async () => {
  let calls = 0;
  await assert.rejects(request('http://example/analyses', { method: 'POST', fetchImpl: async () => { calls++; return response({ error: { code: 'QUEUE_FULL', message: '队列已满' } }, 429); }, sleep: async () => assert.fail('POST retry attempted') }), /队列已满/);
  assert.equal(calls, 1);
});

test('polling preserves task identity and stops at a terminal status', async () => {
  const initial = queued(), statuses = [], bodies = [{ ...initial, status: 'running' }, { ...initial, status: 'succeeded' }];
  const terminal = await pollAnalysis('http://example/api/v1', initial, { fetchImpl: async () => response(bodies.shift()), sleep: async () => {}, onStatus: (e) => statuses.push(e.status) });
  assert.equal(terminal.status, 'succeeded'); assert.deepEqual(statuses, ['queued', 'running', 'succeeded']); assert.equal(bodies.length, 0);
});

test('task failures and expired resources never become success results', async () => {
  const initial = queued();
  await assert.rejects(pollAnalysis('http://example/api/v1', initial, { fetchImpl: async () => response({ ...initial, status: 'failed', error: { code: 'INVALID_AUDIO', message: '音频损坏' } }), sleep: async () => {} }), /音频损坏/);
  let calls = 0;
  await assert.rejects(pollAnalysis('http://example/api/v1', initial, { fetchImpl: async () => { calls++; return response({ error: { message: '已过期' } }, 410); }, sleep: async () => {} }), /已过期/);
  assert.equal(calls, 1);
});

test('task version changes stop polling', async () => {
  const initial = queued();
  await assert.rejects(pollAnalysis('http://example/api/v1', initial, { fetchImpl: async () => response({ ...initial, model: { id: 'mock', version: 'other' } }), sleep: async () => {} }), /来源/);
});

test('client waiting is abortable without sending a server cancellation', async () => {
  const controller = new AbortController();
  const waiting = wait(30000, controller.signal); controller.abort();
  await assert.rejects(waiting, { name: 'AbortError' });
});

test('Retry-After supports seconds, dates, and bounded default backoff', () => {
  assert.equal(retryDelay('3', 0), 3000);
  assert.equal(retryDelay('Thu, 01 Jan 1970 00:00:10 GMT', 0, 1000), 9000);
  assert.equal(retryDelay(null, 4), 5000);
});
