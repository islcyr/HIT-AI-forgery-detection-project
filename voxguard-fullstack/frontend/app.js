import { DEFAULT_LIMITS, validateFile, validateDuration, validateEndpoint, formatTime, timelineLabels, formatScore, normalizeReport, normalizeModels, effectiveLimits, apiURL, previewURL, extractPeaks, createDemoWav } from './analysis.js';
import { request, pollAnalysis } from './api.js';

const $ = (selector) => document.querySelector(selector);
const paths = {
  wave: '<path d="M3 10v4m4-8v12m5-15v18m5-14v10m4-7v4"/>',
  'audio-lines': '<path d="M3 9v6m4-9v12m5-15v18m5-14v10m4-8v6"/>',
  scan: '<path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m13-5v3a2 2 0 0 1-2 2h-3M8 10v4m4-7v10m4-8v6"/>',
  history: '<path d="M3 11a9 9 0 1 1 2.7 7.3M3 4v7h7m2-4v5l3 2"/>',
  sliders: '<path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6m2-6h6m2 8h6"/>',
  book: '<path d="M12 5v16m0-16C9 3 5 3 2 4v16c3-1 7-1 10 1 3-2 7-2 10-1V4c-3-1-7-1-10 1Z"/>',
  'arrow-up-right': '<path d="M7 17 17 7M7 7h10v10"/>',
  'arrow-right': '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  'chevron-right': '<path d="m9 5 7 7-7 7"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.2 9a3 3 0 0 1 5.6 1c0 2-3 2-3 4m.2 3h.01"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
  sparkles: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Zm7 0v4m-2-2h4"/>',
  upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  'file-audio': '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Zm0 0v6h6M8 12v4m4-6v8m4-6v4"/>',
  'shield-check': '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm-4 9 3 3 5-6"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>',
  play: '<path d="m8 4 12 8-12 8V4Z"/>',
  pause: '<path d="M7 4h3v16H7zM14 4h3v16h-3z"/>',
  volume: '<path d="m11 4-6 5H2v6h3l6 5V4Zm4 4a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/>',
  'volume-off': '<path d="m11 4-6 5H2v6h3l6 5V4Zm5 5 5 6m0-6-5 6"/>',
  'mouse-pointer': '<path d="m4 3 6 18 3-7 7-3L4 3Zm9 11 5 5"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  'check-circle': '<circle cx="12" cy="12" r="9"/><path d="m7 12 3 3 7-7"/>',
  headphones: '<path d="M3 14v-3a9 9 0 0 1 18 0v3M3 13h4v8H5a2 2 0 0 1-2-2v-6Zm18 0h-4v8h2a2 2 0 0 0 2-2v-6Z"/>'
};
function icon(name) { return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.info}</svg>`; }
document.querySelectorAll('[data-icon]').forEach((element) => {
  element.innerHTML = icon(element.dataset.icon);
  element.removeAttribute('data-icon');
  if (!element.className) element.replaceWith(element.firstElementChild);
});
document.querySelectorAll('.brand-mark, .upload-symbol, .file-symbol, .dialog-icon').forEach((container) => { container.setAttribute('aria-hidden', 'true'); });

const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const CONFIG_KEY = 'voxguard:v2:config';
const HISTORY_KEY = 'voxguard:v2:history';
const defaultConfig = { endpoint: '/api/v1', modelId: '', mode: 'detect_and_localize' };
function readStorage(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }
const storedConfig = readStorage(CONFIG_KEY, defaultConfig);
const config = { ...defaultConfig, ...(storedConfig && typeof storedConfig === 'object' ? storedConfig : {}) };
try { config.endpoint = validateEndpoint(config.endpoint); } catch { config.endpoint = defaultConfig.endpoint; }
if (!['detect', 'detect_and_localize'].includes(config.mode)) config.mode = 'detect_and_localize';
let history = readStorage(HISTORY_KEY, []);
history = Array.isArray(history) ? history.flatMap((record) => {
  try {
    if (!record || typeof record.fileName !== 'string' || typeof record.fileSize !== 'number') return [];
    return [{ ...record, ...normalizeReport(record.envelope) }];
  } catch { return []; }
}).slice(0, 30) : [];
const audio = $('#audio-player');
const state = { file: null, result: null, duration: 0, peaks: [], buffer: null, url: null, playable: false, loading: false, analyzing: false, controller: null, activeSegment: -1, selection: 0, analysis: 0, isDemoFile: false, models: [], limits: DEFAULT_LIMITS, modelsLoading: false, modelsError: '', modelsGeneration: 0, modelsController: null, previewError: '' };
const selectedModel = () => state.models.find((model) => model.id === config.modelId && model.status === 'ready' && model.capabilities.detection);
const currentLimits = () => effectiveLimits(state.limits, selectedModel());
let toastTimer;

function toast(message) { $('#toast-text').textContent = message; $('#toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 4200); }
function saveStorage(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch { toast('浏览器存储不可用，本次数据仅保留到页面关闭。'); return false; } }
function uploadError(message = '') { $('#upload-error').textContent = message; $('#upload-error').hidden = !message; }
function stopPlayback() { audio.pause(); state.activeSegment = -1; renderActiveSegment(); }
function formatSize(size) { return size >= 1024 * 1024 ? `${(size / 1024 / 1024).toFixed(2)} MiB` : `${Math.round(size / 1024)} KiB`; }
function isPercentageScore(score) { return Number.isFinite(score) && score >= 0 && score <= 1; }
function displayScore(score) { return isPercentageScore(score) ? `${Number((score * 100).toFixed(1))}%` : formatScore(score); }

function updateMode() {
  const model = selectedModel();
  const demo = model?.is_mock === true;
  $('#demo-notice').hidden = !demo;
  $('#mode-tag').classList.toggle('live', Boolean(model && !demo));
  $('#mode-tag').innerHTML = `<span class="status-dot"></span>${model ? demo ? '测试模式' : '音频检测' : '服务未就绪'}`;
  $('#environment-label').textContent = state.modelsLoading ? '正在连接服务' : model ? '检测服务已连接' : '检测服务未就绪';
  $('#environment-detail').textContent = model ? '可以上传音频开始检测' : state.modelsError || '可在接口设置中配置服务';
  $('#privacy-note').textContent = '开始检测后，音频上传至配置的服务；历史记录保存在当前浏览器。';
  $('#api-endpoint').value = config.endpoint;
  const limits = currentLimits();
  $('#upload-limit').textContent = `单文件 · 最大 ${formatSize(limits.max_upload_bytes)}`;
  $('#supported-formats').textContent = `${limits.accepted_formats.map((v) => v.toUpperCase()).join(' · ')} · ${limits.min_duration_sec}～${limits.max_duration_sec} 秒`;
  $('#file-input').accept = limits.accepted_formats.map((v) => `.${v}`).join(',');
  $('#analysis-mode').querySelector('[value="detect_and_localize"]').disabled = !model?.capabilities.localization;
  if (model && !model.capabilities.localization) config.mode = 'detect';
  $('#analysis-mode').value = config.mode;
  $('#analysis-mode').disabled = !model || state.analyzing;
  $('#reload-models').disabled = state.modelsLoading || state.analyzing;
  $('#models-status').textContent = state.modelsLoading ? '正在连接检测服务…' : state.modelsError || (model ? '检测服务已就绪' : '检测服务未就绪，请重新连接。');
  $('#models-status').classList.toggle('connection-error', Boolean(state.modelsError));
}

async function loadModels() {
  state.modelsController?.abort();
  const controller = new AbortController(); state.modelsController = controller;
  const generation = ++state.modelsGeneration;
  state.modelsLoading = true; state.modelsError = ''; state.models = []; updateMode(); updateFileUI();
  try {
    const payload = normalizeModels(await request(apiURL(config.endpoint, 'models'), { signal: controller.signal }));
    if (generation !== state.modelsGeneration) return;
    state.models = payload.models; state.limits = payload.limits;
    config.modelId = payload.models.find((entry) => entry.status === 'ready' && !entry.is_mock && entry.capabilities.detection)?.id || payload.models.find((entry) => entry.status === 'ready' && entry.capabilities.detection)?.id || '';
    if (selectedModel() && !selectedModel().capabilities.localization) config.mode = 'detect';
    saveStorage(CONFIG_KEY, config);
  } catch (error) {
    if (generation !== state.modelsGeneration || controller.signal.aborted) return;
    state.modelsError = error.message;
  } finally {
    if (generation === state.modelsGeneration) { state.modelsLoading = false; state.modelsController = null; updateMode(); updateFileUI(); }
  }
}

const viewLabels = { detect: '语音检测', history: '检测历史', settings: '接口设置', help: '使用指南' };
function route() {
  const view = location.hash.slice(1) in viewLabels ? location.hash.slice(1) : 'detect';
  document.querySelectorAll('.page-view').forEach((element) => { element.hidden = element.id !== `view-${view}`; });
  document.querySelectorAll('.nav-item').forEach((link) => {
    const active = link.dataset.view === view;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
  $('#breadcrumb-current').textContent = viewLabels[view];
  document.title = `VoxGuard · ${viewLabels[view]}`;
  $('#sidebar').classList.remove('open');
  $('#menu-button').setAttribute('aria-expanded', 'false');
  if (view === 'history') renderHistory();
}
window.addEventListener('hashchange', route);
$('#menu-button').addEventListener('click', () => { const open = $('#sidebar').classList.toggle('open'); $('#menu-button').setAttribute('aria-expanded', String(open)); });
document.addEventListener('click', (event) => { if (!event.target.closest('#sidebar, #menu-button') && $('#sidebar').classList.contains('open')) { $('#sidebar').classList.remove('open'); $('#menu-button').setAttribute('aria-expanded', 'false'); } });
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') { $('#sidebar').classList.remove('open'); $('#menu-button').setAttribute('aria-expanded', 'false'); } });

function updateFileUI() {
  const name = state.file?.name || '尚未选择音频';
  $('#file-name').textContent = name;
  $('#file-name').title = name;
  $('#player-file-name').textContent = name;
  $('#player-file-name').title = name;
  const previewState = state.loading ? '正在读取…' : state.playable ? formatTime(state.duration, state.duration < 5) : '本地预览不可用';
  $('#file-meta').textContent = state.file ? `${state.isDemoFile ? '合成测试音（非语音） · ' : ''}${formatSize(state.file.size)} · ${previewState}` : '选择音频后开始检测';
  $('#player-duration').textContent = state.duration ? formatTime(state.duration, state.duration < 5) : '--:--';
  $('#seek-slider').max = state.duration || 1;
  $('#seek-slider').disabled = !state.playable || state.loading;
  $('#play-button').disabled = !state.playable || state.loading;
  $('#analyze-button').disabled = !state.file || state.loading || state.analyzing || !selectedModel() || state.modelsLoading;
  $('#choose-file').disabled = state.analyzing;
  $('#load-demo').disabled = state.analyzing || state.loading;
  $('#analyze-label').textContent = state.loading ? '读取中' : state.analyzing ? '检测中' : state.result ? '重新检测' : '开始检测';
  $('#audio-spec').textContent = state.buffer ? `${state.buffer.sampleRate / 1000} kHz · ${state.buffer.numberOfChannels === 1 ? '单声道' : `${state.buffer.numberOfChannels} 声道`}` : '等待可播放的音频';
}

function riskFor(report) {
  return report.label === 'spoof'
    ? { className: '', label: '疑似伪造', title: '检测到疑似伪造音频', description: '查看下方标记，点击可疑片段试听。' }
    : { className: 'low', label: '未检出伪造', title: '未检测到伪造片段', description: '可以播放音频或导出检测报告。' };
}

function renderResult() {
  const report = state.result;
  const risk = report ? riskFor(report) : { className: 'pending', label: state.analyzing ? '检测中' : '待检测', title: state.analyzing ? '正在分析音频' : '等待音频检测', description: state.analyzing ? '检测完成后展示结果和可疑片段。' : '上传音频后开始检测。' };
  $('#score-value').textContent = report ? displayScore(report.score) : '—';
  $('#score-value').title = report ? String(report.score) : '';
  $('#score-ring').dataset.verdict = report ? report.label : 'pending';
  // Bounded verification scores use a percentage scale, not a probability claim.
  const ringFraction = report ? isPercentageScore(report.score) ? report.score : 1 : 0;
  $('#ring-value').style.strokeDashoffset = String(477.522 * (1 - ringFraction));
  $('#risk-badge').className = `risk-badge ${risk.className}`;
  $('#risk-badge').innerHTML = `<span class="status-dot"></span>${risk.label}`;
  $('#result-summary').textContent = risk.title;
  $('#result-description').textContent = risk.description;
  $('#result-state').innerHTML = `<span class="status-dot"></span>${report ? '分析完成' : state.analyzing ? '分析进行中' : '等待音频检测'}`;
  $('#result-demo-label').hidden = !report?.envelope.is_mock;
  $('#result-demo-label').textContent = report?.envelope.is_mock ? '测试模式' : '';
  $('#result-demo-label').classList.toggle('mock-result', report?.envelope.is_mock === true);
  const localized = report?.localizationStatus === 'available';
  $('#suspicious-duration').innerHTML = localized ? `${report.suspiciousDuration.toFixed(2)} <small>秒</small>` : '—';
  $('#segment-count').innerHTML = localized ? `${report.segments.length} <small>处</small>` : '—';
  $('#total-duration').innerHTML = state.duration ? `${state.duration.toFixed(2)} <small>秒</small>` : '—';
  $('#segments-heading-count').textContent = localized ? String(report.segments.length).padStart(2, '0') : '—';
  $('#export-report').disabled = !report;
  $('#result-metadata').textContent = report ? `检测完成于 ${new Date(report.envelope.completed_at).toLocaleString('zh-CN')}` : '';
  $('#result-warnings').textContent = report ? [state.previewError, ...report.envelope.warnings.map((item) => item.message)].filter(Boolean).join('；') : '';
  $('#result-warnings').hidden = !$('#result-warnings').textContent;
  renderWaveform(); renderSegments(); updateFileUI();
}

function renderWaveform() {
  const duration = state.duration || 1;
  const segments = state.result?.segments || [];
  let svg = '<line x1="0" x2="1200" y1="56" y2="56" stroke="#f2edf9"/>';
  segments.forEach((segment) => {
    const start = segment.start / duration * 1200;
    const width = (segment.end - segment.start) / duration * 1200;
    svg += `<rect x="${start}" y="0" width="${width}" height="112" fill="#fcf0f3"/><line x1="${start}" x2="${start}" y1="0" y2="112" stroke="#edc3cd" stroke-dasharray="3 3"/><line x1="${start + width}" x2="${start + width}" y1="0" y2="112" stroke="#edc3cd" stroke-dasharray="3 3"/>`;
  });
  const peaks = state.peaks.length ? state.peaks : Array(300).fill(0);
  const maximum = Math.max(...peaks, .001);
  peaks.forEach((peak, index) => {
    const time = (index + .5) / peaks.length * duration;
    const suspicious = segments.some((segment) => time >= segment.start && time <= segment.end);
    const height = Math.max(3, Math.pow(peak / maximum, .7) * 90);
    const width = 1200 / peaks.length;
    svg += `<rect x="${index * width}" y="${56 - height / 2}" width="${Math.max(1, width - 1.8)}" height="${height}" rx="1" fill="${suspicious ? '#dfa0af' : '#b9a5ed'}"/>`;
  });
  $('#waveform-svg').innerHTML = svg;
  $('#segment-markers').innerHTML = segments.map((segment, index) => `<button class="segment-marker" style="left:${Math.min(96, segment.start / duration * 100)}%" data-segment="${index}" aria-label="试听可疑片段 ${index + 1}">${String(index + 1).padStart(2, '0')}</button>`).join('');
  $('#timeline-labels').innerHTML = timelineLabels(state.duration).map((label) => `<span>${label}</span>`).join('');
  updatePlaybackPosition();
}

function renderSegments() {
  const report = state.result;
  const segments = report?.segments || [];
  const message = !report ? state.analyzing ? '分析完成后，可疑位置将在这里展示。' : '开始检测后查看定位结果。'
    : report.localizationStatus === 'unsupported' ? '本次检测仅提供整段结果。'
    : report.localizationStatus === 'not_requested' ? '本次仅请求整段检测，未执行时间片段定位。'
    : '未检测到可疑片段。';
  $('#segment-list').innerHTML = segments.length ? segments.map((segment, index) => `<button class="segment-item" data-segment="${index}" ${state.playable ? '' : 'disabled'} aria-label="试听可疑片段 ${index + 1}，${formatTime(segment.start, true)} 至 ${formatTime(segment.end, true)}，检测分数 ${displayScore(segment.score)}"><span class="segment-top"><span class="segment-index">${String(index + 1).padStart(2, '0')}</span>可疑片段 ${index + 1}${icon('play')}</span><strong class="segment-time">${formatTime(segment.start, true)} <span>–</span> ${formatTime(segment.end, true)}</strong><span class="segment-bottom"><span>${(segment.end - segment.start).toFixed(2)} 秒</span><strong title="${segment.score}">${displayScore(segment.score)}</strong></span></button>`).join('') : `<p class="no-segments">${message}</p>`;
  renderActiveSegment();
}

function renderActiveSegment() { document.querySelectorAll('.segment-item').forEach((element) => { const active = Number(element.dataset.segment) === state.activeSegment; element.classList.toggle('active', active); element.setAttribute('aria-pressed', String(active)); }); }
function updatePlaybackPosition() {
  const time = Math.min(audio.currentTime || 0, state.duration || 0);
  $('#current-time').textContent = formatTime(time, state.duration < 5);
  $('#seek-slider').value = time;
  $('#seek-slider').setAttribute('aria-valuetext', `${formatTime(time, true)}，共 ${formatTime(state.duration, state.duration < 5)}`);
  $('#playhead').style.left = `${state.duration ? Math.min(100, time / state.duration * 100) : 0}%`;
}
async function playSegment(index) {
  const segment = state.result?.segments?.[index];
  if (!segment) return;
  if (!state.playable) { toast('当前音频预览不可用，请检查预览资源是否仍有效。'); return; }
  audio.currentTime = segment.start;
  state.activeSegment = index;
  renderActiveSegment();
  updatePlaybackPosition();
  try { await audio.play(); } catch { toast('音频无法播放，请检查文件或浏览器的音频权限。'); }
}
$('#segment-list').addEventListener('click', (event) => { const item = event.target.closest('[data-segment]'); if (item) playSegment(Number(item.dataset.segment)); });
$('#segment-markers').addEventListener('click', (event) => { const item = event.target.closest('[data-segment]'); if (item) playSegment(Number(item.dataset.segment)); });
$('#play-button').addEventListener('click', async () => {
  if (audio.paused) {
    state.activeSegment = -1; renderActiveSegment();
    if (audio.currentTime >= state.duration - .05) audio.currentTime = 0;
    try { await audio.play(); } catch { toast('音频无法播放，请检查文件或浏览器的音频权限。'); }
  } else stopPlayback();
});
audio.addEventListener('play', () => { $('#play-button').innerHTML = icon('pause'); $('#play-button').setAttribute('aria-label', '暂停音频'); });
audio.addEventListener('pause', () => { $('#play-button').innerHTML = icon('play'); $('#play-button').setAttribute('aria-label', '播放音频'); });
audio.addEventListener('ended', () => { state.activeSegment = -1; renderActiveSegment(); });
audio.addEventListener('timeupdate', () => {
  updatePlaybackPosition();
  const segment = state.result?.segments?.[state.activeSegment];
  if (segment && audio.currentTime >= segment.end) { stopPlayback(); audio.currentTime = segment.end; updatePlaybackPosition(); }
});
audio.addEventListener('loadedmetadata', () => {
  if (!state.loading && !state.result && Number.isFinite(audio.duration) && audio.duration > 0) {
    state.duration = audio.duration; state.playable = true; updateFileUI(); renderResult();
  }
});
$('#seek-slider').addEventListener('input', (event) => { state.activeSegment = -1; renderActiveSegment(); audio.currentTime = Number(event.target.value); updatePlaybackPosition(); });
$('#speed-button').addEventListener('click', () => { const speeds = [1, 1.25, 1.5, 2, .75]; audio.playbackRate = speeds[(speeds.indexOf(audio.playbackRate) + 1) % speeds.length]; $('#speed-button').textContent = `${audio.playbackRate}×`; $('#speed-button').setAttribute('aria-label', `播放速度，当前 ${audio.playbackRate} 倍`); });
$('#mute-button').addEventListener('click', () => { audio.muted = !audio.muted; $('#mute-button').innerHTML = icon(audio.muted ? 'volume-off' : 'volume'); $('#mute-button').setAttribute('aria-label', audio.muted ? '取消静音' : '静音'); });

function cancelAnalysis(notify = true) {
  if (!state.analyzing) return;
  state.analysis++;
  state.controller?.abort(); state.controller = null; state.analyzing = false;
  if (state.loading) { state.selection++; clearAudio(); state.loading = false; if (state.result) state.previewError = '已停止加载音频预览，检测结果已保留。'; }
  $('#analysis-progress').hidden = true;
  $('#dropzone').classList.remove('loading');
  renderResult();
  updateMode();
  if (notify) toast('已停止本页面等待，不表示服务端任务已取消。');
}
$('#cancel-analysis').addEventListener('click', () => cancelAnalysis());

function clearAudio() {
  stopPlayback();
  if (state.url) URL.revokeObjectURL(state.url);
  audio.removeAttribute('src'); audio.load();
  state.url = null; state.buffer = null; state.peaks = []; state.playable = false;
}

async function decodeBlob(blob, selection, canonicalDuration = null) {
  let context;
  try {
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) throw new Error('当前浏览器不支持波形解码。');
    context = new Context();
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    if (selection !== state.selection) return;
    if (canonicalDuration !== null && Math.abs(buffer.duration - canonicalDuration) > 0.05) throw new Error('服务端预览时长与定位时间轴不一致，已禁用播放。');
    state.buffer = buffer; state.duration = canonicalDuration ?? buffer.duration; state.peaks = extractPeaks(buffer); state.playable = true;
  } finally { if (context) await context.close().catch(() => {}); }
}

async function selectFile(file, isDemoFile = false) {
  if (!file || state.analyzing) return false;
  try { validateFile(file, currentLimits()); } catch (error) { uploadError(error.message); return false; }
  cancelAnalysis(false); clearAudio(); uploadError();
  const selection = ++state.selection;
  state.file = file; state.isDemoFile = isDemoFile; state.result = null; state.previewError = ''; state.duration = 0; state.loading = true;
  state.url = URL.createObjectURL(file); audio.src = state.url; audio.load(); renderResult();
  try { await decodeBlob(file, selection); }
  catch {
    if (selection !== state.selection) return false;
    if (Number.isFinite(audio.duration) && audio.duration > 0) { state.duration = audio.duration; state.playable = true; }
    else uploadError('本地预览不可用，仍可提交检测；成功后将加载服务端 WAV 预览。');
  }
  if (selection !== state.selection) return false;
  if (state.duration) { try { validateDuration(state.duration, currentLimits()); } catch (error) { uploadError(error.message); } }
  state.loading = false; renderResult(); return true;
}

$('#choose-file').addEventListener('click', () => $('#file-input').click());
$('#file-input').addEventListener('change', async (event) => { await selectFile(event.target.files[0]); event.target.value = ''; });
let dragDepth = 0;
$('#dropzone').addEventListener('dragenter', (event) => { event.preventDefault(); if (!state.analyzing) { dragDepth++; $('#dropzone').classList.add('drag-over'); } });
$('#dropzone').addEventListener('dragover', (event) => { event.preventDefault(); event.dataTransfer.dropEffect = state.analyzing ? 'none' : 'copy'; });
$('#dropzone').addEventListener('dragleave', (event) => { event.preventDefault(); if (--dragDepth <= 0) { dragDepth = 0; $('#dropzone').classList.remove('drag-over'); } });
$('#dropzone').addEventListener('drop', (event) => { event.preventDefault(); dragDepth = 0; $('#dropzone').classList.remove('drag-over'); if (state.analyzing) return; if (event.dataTransfer.files.length !== 1) { uploadError('请一次上传一个音频文件。'); return; } selectFile(event.dataTransfer.files[0]); });
// Prevent dropped files outside the upload area from navigating away from the app.
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());

function progress(value, message, status = '等待响应') { $('#progress-fill').style.width = `${value}%`; $('#progress-percent').textContent = status; $('#progress-text').textContent = message; }

async function runAnalysis() {
  const model = selectedModel();
  if (!state.file || state.loading || state.analyzing || !model) return;
  try { validateFile(state.file, currentLimits()); if (state.duration) validateDuration(state.duration, currentLimits()); }
  catch (error) { uploadError(error.message); return; }
  stopPlayback(); uploadError();
  state.analyzing = true; state.result = null; state.previewError = ''; state.controller = new AbortController();
  const controller = state.controller, attempt = ++state.analysis, selection = state.selection;
  const file = state.file, base = config.endpoint, mode = config.mode;
  $('#analysis-progress').hidden = false; $('#dropzone').classList.add('loading');
  progress(12, '正在提交音频…', '上传中'); updateMode(); renderResult();
  try {
    const form = new FormData(); form.append('file', file, file.name); form.append('model_id', model.id); form.append('mode', mode); form.append('include_timeline', 'false');
    const initial = await request(apiURL(base, 'analyses'), { method: 'POST', body: form, signal: controller.signal });
    if (initial.model?.id !== model.id || initial.mode !== mode) throw new Error('服务返回的模型或任务模式与请求不一致。');
    const envelope = await pollAnalysis(base, initial, { signal: controller.signal, onStatus: (task, elapsed) => {
      if (attempt !== state.analysis) return;
      const queued = task.status === 'queued';
      progress(queued ? 28 : 65, `${queued ? '任务已接受，等待分析' : '正在执行音频分析'}${elapsed > 120000 ? '；处理时间较长，可继续等待或停止本页面等待' : ''}。`, queued ? '排队中' : '分析中');
    } });
    if (attempt !== state.analysis) return;
    const report = normalizeReport(envelope);
    state.result = { ...report, fileName: file.name, fileSize: file.size, apiBase: base };
    state.duration = report.duration;
    history = [state.result, ...history.filter((entry) => entry.envelope.analysis_id !== envelope.analysis_id)].slice(0, 30);
    saveStorage(HISTORY_KEY, history); renderHistory();
    clearAudio(); state.duration = report.duration; state.loading = true; renderResult();
    progress(92, '分析成功，正在加载 WAV 预览…', '加载预览');
    try {
      const blob = await request(previewURL(base, envelope.audio.preview_url), { signal: controller.signal, binary: true });
      if (attempt !== state.analysis || selection !== state.selection) return;
      state.url = URL.createObjectURL(blob); audio.src = state.url; audio.load();
      await decodeBlob(blob, selection, report.duration);
    } catch (error) {
      if (attempt !== state.analysis || controller.signal.aborted) return;
      clearAudio(); state.previewError = `检测结果已保留；预览不可用：${error.message}`;
    }
    if (attempt !== state.analysis) return;
    toast(envelope.is_mock ? '模拟联调完成，分数和片段不代表真实模型结果。' : '检测完成，结果已保存在本机历史。');
  } catch (error) {
    if (attempt !== state.analysis || controller.signal.aborted) return;
    uploadError(error.message);
  } finally {
    if (attempt === state.analysis) { state.loading = false; state.analyzing = false; state.controller = null; $('#analysis-progress').hidden = true; $('#dropzone').classList.remove('loading'); updateMode(); renderResult(); }
  }
}
$('#analyze-button').addEventListener('click', runAnalysis);
$('#load-demo').addEventListener('click', async () => {
  if (await selectFile(createDemoWav(), true)) toast('已载入合成测试音（非语音）。点击开始检测后才会向服务提交任务。');
});
$('#reload-models').addEventListener('click', loadModels);
$('#analysis-mode').addEventListener('change', (event) => { config.mode = event.target.value; saveStorage(CONFIG_KEY, config); });

function downloadReport(report) {
  const data = { product: 'VoxGuard', version: '2.0', notice: report.envelope.is_mock ? '模拟联调结果，不能用于真实性判断。' : '模型分数未经概率校准，仅供辅助核验。', file: { name: report.fileName, size_bytes: report.fileSize }, analysis: report.envelope };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url;
  anchor.download = `VoxGuard_${report.fileName.replace(/\.[^.]+$/, '').replace(/[<>:"/\\|?*\x00-\x1F]/g, '_').slice(0, 80)}_${report.envelope.is_mock ? 'mock' : 'report'}.json`;
  document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
  toast('报告已导出为 JSON 文件。');
}
$('#export-report').addEventListener('click', () => { if (state.result) downloadReport(state.result); });

function renderHistory() {
  $('#history-count').textContent = history.length;
  $('#history-empty').hidden = history.length > 0;
  $('.history-table-wrap').hidden = history.length === 0;
  $('#clear-history').disabled = history.length === 0;
  $('#history-body').innerHTML = history.map((record) => `<tr><td><strong class="history-file-name" title="${escapeHTML(record.fileName)}">${escapeHTML(record.fileName)}</strong><span class="history-mode">${formatTime(record.duration, true)}</span></td><td><span class="history-score" title="${record.score}">${displayScore(record.score)}</span><span class="history-mode">${record.label === 'spoof' ? '疑似伪造' : '未检出伪造'}</span></td><td>${record.localizationStatus === 'available' ? `${record.segments.length} 处` : record.localizationStatus === 'unsupported' ? '不支持定位' : '未请求定位'}</td><td>${escapeHTML(new Date(record.envelope.completed_at).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }))}</td><td><button class="history-export" data-report-id="${escapeHTML(record.envelope.analysis_id)}" aria-label="导出 ${escapeHTML(record.fileName)} 的检测报告">${icon('download')}导出</button></td></tr>`).join('');
}
$('#history-body').addEventListener('click', (event) => { const button = event.target.closest('[data-report-id]'); if (button) { const report = history.find((record) => record.envelope.analysis_id === button.dataset.reportId); if (report) downloadReport(report); } });
$('#clear-history').addEventListener('click', () => $('#confirm-dialog').showModal());
$('#cancel-clear').addEventListener('click', () => $('#confirm-dialog').close());
$('#confirm-clear').addEventListener('click', () => { history = []; saveStorage(HISTORY_KEY, history); renderHistory(); $('#confirm-dialog').close(); toast('本机检测历史已清空。'); });

$('#settings-form').addEventListener('submit', (event) => {
  event.preventDefault(); $('#settings-error').hidden = true;
  let endpoint;
  try { endpoint = validateEndpoint($('#api-endpoint').value); } catch (error) { $('#settings-error').textContent = error.message; $('#settings-error').hidden = false; $('#api-endpoint').focus(); return; }
  cancelAnalysis(false);
  const changed = config.endpoint !== endpoint;
  config.endpoint = endpoint; saveStorage(CONFIG_KEY, config);
  if (changed) { state.result = null; state.previewError = ''; uploadError(); }
  renderResult(); loadModels(); toast('接口设置已保存，正在重新连接检测服务。');
});

window.addEventListener('beforeunload', () => { state.controller?.abort(); state.modelsController?.abort(); if (state.url) URL.revokeObjectURL(state.url); });
updateMode(); renderHistory(); renderResult(); route(); loadModels();
