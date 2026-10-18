export const DEFAULT_LIMITS = Object.freeze({ accepted_formats: ['wav', 'flac', 'mp3'], max_upload_bytes: 20971520, min_duration_sec: 0.5, max_duration_sec: 120 });

const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const positive = (value) => finite(value) && value > 0;
const text = (value) => typeof value === 'string' && value.length > 0;

export function validateFile(file, limits = DEFAULT_LIMITS) {
  const extension = file.name.split('.').pop().toLowerCase();
  if (!limits.accepted_formats.includes(extension)) throw new Error(`请选择 ${limits.accepted_formats.map((v) => v.toUpperCase()).join('、')} 音频文件。`);
  if (file.size === 0) throw new Error('文件为空，请重新选择音频。');
  if (file.size > limits.max_upload_bytes) throw new Error(`文件超过 ${(limits.max_upload_bytes / 1048576).toFixed(1)} MiB 的上传限制。`);
}

export function validateDuration(duration, limits = DEFAULT_LIMITS) {
  if (!positive(duration) || duration < limits.min_duration_sec || duration > limits.max_duration_sec) {
    throw new Error(`音频时长须为 ${limits.min_duration_sec}～${limits.max_duration_sec} 秒；最终以服务端解码结果为准。`);
  }
}

export function formatTime(seconds, precision = false) {
  const safe = finite(seconds) ? Math.max(0, seconds) : 0;
  const ticks = precision ? Math.round(safe * 10) : Math.floor(safe) * 10;
  return `${String(Math.floor(ticks / 600)).padStart(2, '0')}:${String(Math.floor(ticks / 10) % 60).padStart(2, '0')}${precision ? `.${ticks % 10}` : ''}`;
}

export function formatScore(value) { return finite(value) ? String(Number(value.toPrecision(6))) : '—'; }

export function timelineLabels(duration) {
  if (!positive(duration)) return [];
  if (duration < 0.1) return ['00:00', `${duration.toFixed(3)} 秒`];
  const count = Math.min(6, Math.floor(duration * 10) + 1);
  return Array.from({ length: count }, (_, i) => formatTime(duration * i / (count - 1), duration < 5));
}

export function validateEndpoint(value) {
  const endpoint = value.trim().replace(/\/+$/, '');
  if (endpoint.startsWith('/') && !endpoint.startsWith('//') && !/[\\?#]/.test(endpoint)) return endpoint;
  try {
    const url = new URL(endpoint);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error();
    return url.href.replace(/\/+$/, '');
  } catch { throw new Error('请填写 HTTP(S) API 基础地址或 /api/v1 形式的同源路径，不含查询参数。'); }
}

export function apiURL(base, path, pageURL = globalThis.location?.href) {
  return new URL(`${validateEndpoint(base)}/${path.replace(/^\/+/, '')}`, pageURL).href;
}

export function previewURL(base, preview, pageURL = globalThis.location?.href) {
  const root = new URL(`${validateEndpoint(base)}/`, pageURL);
  const url = new URL(preview, root);
  if (url.origin !== root.origin || !['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('预览地址必须与 API 服务同源。');
  return url.href;
}

export function normalizeModels(payload) {
  const limits = payload?.limits;
  if (!Array.isArray(payload?.models) || !limits || !Array.isArray(limits.accepted_formats) || !limits.accepted_formats.length || !limits.accepted_formats.every((v) => ['wav', 'flac', 'mp3'].includes(v)) || !positive(limits.max_upload_bytes) || !positive(limits.min_duration_sec) || !positive(limits.max_duration_sec) || limits.max_duration_sec < limits.min_duration_sec) throw new Error('模型目录或服务输入限制格式无效。');
  const ids = new Set();
  for (const model of payload.models) {
    if (!text(model.id) || ids.has(model.id) || !text(model.version) || !text(model.display_name) || typeof model.is_mock !== 'boolean' || !['ready', 'unavailable'].includes(model.status) || typeof model.capabilities?.detection !== 'boolean' || typeof model.capabilities?.localization !== 'boolean' || !positive(model.input?.min_duration_sec) || !positive(model.input?.max_duration_sec) || model.input.max_duration_sec < model.input.min_duration_sec) throw new Error('模型目录缺少能力、版本或模拟标识。');
    ids.add(model.id);
  }
  return payload;
}

export function effectiveLimits(serviceLimits = DEFAULT_LIMITS, model) {
  return { ...serviceLimits, min_duration_sec: Math.max(serviceLimits.min_duration_sec, model?.input.min_duration_sec ?? 0), max_duration_sec: Math.min(serviceLimits.max_duration_sec, model?.input.max_duration_sec ?? Infinity) };
}

export function validateEnvelope(payload) {
  if (!payload || !text(payload.analysis_id) || !['queued', 'running', 'succeeded', 'failed'].includes(payload.status) || !['detect', 'detect_and_localize'].includes(payload.mode) || !text(payload.model?.id) || !text(payload.model?.version) || typeof payload.is_mock !== 'boolean' || !text(payload.created_at) || !text(payload.expires_at) || !Array.isArray(payload.warnings)) throw new Error('任务响应格式无效或缺少模型来源信息。');
  if (payload.status === 'failed' && !text(payload.error?.message)) throw new Error('失败任务缺少错误信息。');
  if (['queued', 'running'].includes(payload.status) && (payload.result !== null || payload.error !== null)) throw new Error('任务状态与结果不一致。');
  return payload;
}

export function normalizeReport(payload) {
  validateEnvelope(payload);
  if (payload.status !== 'succeeded') throw new Error('任务尚未成功，不能展示检测结论。');
  const duration = payload.audio?.duration_sec;
  const decision = payload.result?.decision;
  const localization = payload.result?.localization;
  if (!positive(duration) || !text(payload.audio.preview_url) || !text(payload.completed_at) || payload.error !== null || !['bonafide', 'spoof'].includes(decision?.label) || !finite(decision.spoof_score) || !finite(decision.threshold) || decision.score_direction !== 'higher_is_spoof' || decision.score_type !== 'uncalibrated') throw new Error('检测结果中的时长、分数、阈值或分数语义无效。');
  if (!['available', 'unsupported', 'not_requested'].includes(localization?.status)) throw new Error('定位状态无效。');
  let segments = [];
  if (localization.status === 'available') {
    if (!Array.isArray(localization.segments) || !positive(localization.resolution_sec) || !finite(localization.threshold) || !text(localization.postprocessing_version)) throw new Error('定位结果缺少片段或后处理信息。');
    let previousEnd = 0;
    segments = localization.segments.map((segment) => {
      if (!finite(segment.start_sec) || !finite(segment.end_sec) || segment.start_sec < previousEnd || segment.start_sec < 0 || segment.end_sec <= segment.start_sec || segment.end_sec > duration || !finite(segment.spoof_score)) throw new Error('定位片段须按时间排序、不重叠且位于原始音频范围内。');
      previousEnd = segment.end_sec;
      return { start: segment.start_sec, end: segment.end_sec, score: segment.spoof_score };
    });
  } else if (localization.segments !== null) throw new Error('未执行定位时，segments 须为 null。');
  if (!['gte', 'gt'].includes(decision.threshold_operator ?? 'gte')) throw new Error('检测阈值比较规则无效。');
  if (!['timeline_mean', 'proposal'].includes(localization.score_aggregation ?? 'timeline_mean')) throw new Error('定位分数类型无效。');
  return { thresholdOperator: decision.threshold_operator ?? 'gte', sourceScoreType: decision.source_score_type ?? null, scoreAggregation: localization.score_aggregation ?? 'timeline_mean', score: decision.spoof_score, threshold: decision.threshold, label: decision.label, duration, localizationStatus: localization.status, segments, suspiciousDuration: localization.status === 'available' ? segments.reduce((sum, s) => sum + s.end - s.start, 0) : null, envelope: payload };
}

export function extractPeaks(buffer, count = 300) {
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) => buffer.getChannelData(index));
  const step = Math.max(1, Math.floor(buffer.length / count));
  return Array.from({ length: count }, (_, index) => {
    let max = 0;
    for (let offset = index * step; offset < Math.min((index + 1) * step, buffer.length); offset += Math.max(1, Math.floor(step / 110))) {
      for (const channel of channels) max = Math.max(max, Math.abs(channel[offset]));
    }
    return max;
  });
}

// A locally generated, test tone. It is not a recording or a model prediction.
export function createDemoWav() {
  const rate = 16000;
  const duration = 24;
  const length = rate * duration;
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  writeText(0, 'RIFF'); view.setUint32(4, 36 + length * 2, true); writeText(8, 'WAVE');
  writeText(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); writeText(36, 'data'); view.setUint32(40, length * 2, true);
  let phase = 0;
  let seed = 194;
  for (let index = 0; index < length; index++) {
    const t = index / rate;
    const envelope = Math.pow(Math.max(0, Math.sin(t * 5.1) * .58 + Math.sin(t * 1.9) * .23 + .26), 1.5);
    const pause = (t % 4.3 > 3.6) ? .06 : 1;
    phase += 2 * Math.PI * (125 + 17 * Math.sin(t * 2.5)) / rate;
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const noise = ((seed / 4294967296) * 2 - 1) * .03;
    const signal = (Math.sin(phase) * .35 + Math.sin(phase * 3) * .15 + Math.sin(phase * 7) * .08 + noise) * envelope * pause;
    const fade = Math.min(1, t * 8, (duration - t) * 8);
    view.setInt16(44 + index * 2, Math.max(-1, Math.min(1, signal * fade)) * 32767, true);
  }
  return new File([buffer], 'synthetic_test_tone.wav', { type: 'audio/wav' });
}
