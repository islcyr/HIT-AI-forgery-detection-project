import { apiURL, validateEnvelope } from './analysis.js';

export function wait(ms, signal) {
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(signal.reason ?? new DOMException('已停止等待', 'AbortError')); };
    const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, ms);
    if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, { once: true });
  });
}

export function retryDelay(header, attempt, now = Date.now()) {
  const numeric = header === null || header.trim() === '' ? NaN : Number(header);
  const parsed = Number.isFinite(numeric) ? numeric * 1000 : Date.parse(header) - now;
  return Number.isFinite(parsed) ? Math.max(0, parsed) : Math.min(5000, 1000 * 2 ** attempt);
}

// Only GET requests are retried. Uploads lack an idempotency guarantee.
export async function request(url, { method = 'GET', body, signal, binary = false, fetchImpl = globalThis.fetch, sleep = wait } = {}) {
  for (let attempt = 0; ; attempt++) {
    const controller = new AbortController();
    const abort = () => controller.abort(signal.reason);
    if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, { once: true });
    const timeout = setTimeout(() => controller.abort(new DOMException('请求超时', 'TimeoutError')), method === 'POST' ? 120000 : 30000);
    let response, payload;
    try {
      response = await fetchImpl(url, { method, body, signal: controller.signal, credentials: 'omit', headers: { Accept: binary ? 'audio/wav, application/json' : 'application/json' } });
      if (response.ok && binary) return await response.blob();
      try { payload = await response.json(); } catch {
        if (response.ok) throw new Error(`服务返回的内容不是有效 JSON（HTTP ${response.status}）。`);
        // A proxy may return an HTML error; the HTTP retry policy still applies.
        payload = null;
      }
    } catch (error) {
      if (signal?.aborted) throw signal.reason ?? error;
      if (controller.signal.reason?.name === 'TimeoutError') throw new Error(method === 'POST' ? '上传请求超时，服务端可能已创建任务。未自动重复上传；请检查服务状态。' : '查询超时。已停止本次等待，服务端任务状态未改变。');
      if (error instanceof TypeError) throw new Error(method === 'POST' ? '上传连接中断，服务端可能已创建任务。未自动重复上传；请检查网络、服务及跨域配置。' : '无法连接服务，请检查网络、服务地址及跨域配置。');
      throw error;
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', abort);
    }
    if (response.ok) return payload;
    const delay = retryDelay(response.headers.get('Retry-After'), attempt);
    if (method === 'GET' && [429, 503].includes(response.status) && attempt < 3 && delay <= 30000) {
      await sleep(delay, signal);
      continue;
    }
    const message = typeof payload?.error?.message === 'string' ? payload.error.message : `服务返回 HTTP ${response.status}。`;
    const error = new Error(`${message}${[429, 503].includes(response.status) ? ' 请稍后重试。' : ''}`);
    error.status = response.status;
    error.code = payload?.error?.code;
    throw error;
  }
}

export async function pollAnalysis(base, initial, { signal, onStatus = () => {}, fetchImpl, sleep = wait } = {}) {
  let envelope = validateEnvelope(initial);
  const started = Date.now();
  while (true) {
    onStatus(envelope, Date.now() - started);
    if (envelope.status === 'succeeded') return envelope;
    if (envelope.status === 'failed') throw new Error(`${envelope.error.message}（${envelope.error.code ?? 'TASK_FAILED'}）`);
    await sleep(1000, signal);
    const next = validateEnvelope(await request(apiURL(base, `analyses/${encodeURIComponent(initial.analysis_id)}`), { signal, fetchImpl, sleep }));
    if (next.analysis_id !== initial.analysis_id || next.model.id !== initial.model.id || next.model.version !== initial.model.version || next.is_mock !== initial.is_mock || next.mode !== initial.mode) throw new Error('任务标识或模型来源在查询期间发生变化，已停止展示。');
    if (envelope.status === 'running' && next.status === 'queued') throw new Error('任务状态回退，已停止查询。');
    envelope = next;
  }
}
