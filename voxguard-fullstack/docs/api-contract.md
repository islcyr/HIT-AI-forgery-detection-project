# 音频检测前后端接口约定

v0.2 工程约定单文件上传、异步检测和结果播放，不依赖具体后端框架。字段定义以 [OpenAPI](../contracts/openapi.yaml) 为准；本文件说明流程和业务语义。

## 用户操作与接口

| 顺序 | 接口 | 用途 |
| --- | --- | --- |
| 1 | `GET /api/v1/models` | 查询模型、可用状态、是否支持定位及上传限制 |
| 2 | `POST /api/v1/analyses` | 上传一个音频文件，创建分析任务，成功返回 `202` |
| 3 | `GET /api/v1/analyses/{analysis_id}` | 查询状态；成功后取得整段判断和时间区间 |
| 4 | `GET /api/v1/analyses/{analysis_id}/audio` | 任务成功后取得 WAV 预览音频，供播放及绘制波形 |

请求路径中的 ID 为后端生成的不透明标识，不使用用户文件名。模型列表只把 `ready` 的模型作为可选项，并通过 `is_mock` 显著区分模拟适配器和真实模型；完成检测和定位需要 `capabilities.localization=true`。

上传使用 `multipart/form-data`：

| 字段 | 是否必填 | 约定 |
| --- | --- | --- |
| `file` | 是 | 单个 WAV、FLAC 或 MP3 文件，以实际解码结果确认格式 |
| `model_id` | 是 | 来自模型列表；后端在创建任务时固定所用模型版本 |
| `mode` | 否 | 默认 `detect_and_localize`；仅检测为 `detect` |
| `include_timeline` | 否 | 默认 `false`；只有定位模式允许 `true`，用于获取分数时间序列 |

建议服务限制为 20 MiB，即 20,971,520 字节，时长 0.5～120 秒，单声道或双声道。实际约束取服务限制与所选模型输入限制的交集。采样率由后端按模型需要转换，必须对采样数据执行重采样，不得仅修改文件头。上述限制为产品建议，不代表所有模型的固有输入限制。

## 任务状态与前端显示

通常状态为 `queued → running → succeeded` 或 `failed`；排队期间也可因模型不可用等原因进入 `failed`。客户端可能直接看到终态。状态不会从终态退回；同一 ID 的终态结果保持不变。

- `queued`：显示“等待分析”，`result=null`、`error=null`、`completed_at=null`。
- `running`：显示“正在分析”，`result=null`、`error=null`、`completed_at=null`。
- `succeeded`：`audio`、`result`、`completed_at` 有值，`error=null`。
- `failed`：`result=null`，`error`、`completed_at` 有值；显示错误信息，不能显示“真实”。

建议每秒查询一次，遇到 `429` 或 `503` 按 `Retry-After` 等待，无该头时逐步退避到 5 秒。终态及 `404/410` 停止查询；页面关闭或切换任务时取消查询。等待超过 2 分钟可提示仍在处理，不得仅因前端超时将服务端任务判定为失败。上传没有幂等保证，网络错误后不得自动重复 POST。

查询一个已经执行失败的任务仍返回 HTTP `200`，任务体为 `status=failed`；HTTP 错误用于请求本身失败。建议上传和结果保留到 `expires_at`，默认创建后 24 小时。过期任务返回 `410`，未知 ID 返回 `404`；过期识别所需的最小记录可以保留。重试分析使用新任务 ID。

## 整段检测分数

`decision.label` 为 `bonafide` 或 `spoof`。统一把分数方向转为 `higher_is_spoof`，默认 `threshold_operator=gte`，当 `spoof_score >= threshold` 时判为 `spoof`；`gt` 使用严格大于，须与模型描述的 `decision_operator` 一致。`source_score_type` 和 `decision_rule` 可保留原生分数及判定语义。分数是有限实数，不限定为 0～1；`score_type=uncalibrated` 表示未承诺概率校准。

前端用“疑似伪造”或“当前模型未检出伪造”表述，分数如展示需标为“模型分数”。不同模型的分数不能直接排名。整体检测阈值与定位阈值可以不同，均绑定在结果引用的不可变模型版本中。

## 时间片段与定位状态

所有时间均为秒，相对于原始上传音频起点。区间为左闭右开 `[start_sec,end_sec)`，满足 `0 ≤ start_sec < end_sec ≤ audio.duration_sec`；按开始时间排序、不重叠。返回区间代表模型预测，不代表数据集真值。

| `localization.status` | `segments` | 前端显示 |
| --- | --- | --- |
| `available` | 一个或多个区间 | 标红这些疑似伪造区间 |
| `available` | `[]` | 已完成定位，当前阈值下未发现疑似片段 |
| `unsupported` | `null` | 当前模型不支持定位；只出现在该模型的 `detect` 模式 |
| `not_requested` | `null` | 模型支持定位，但本次只请求整段检测 |

后两种状态的 `resolution_sec`、`threshold`、`postprocessing_version` 都为 `null`，`timeline=null`。请求 `detect_and_localize` 时若模型不支持定位，创建阶段返回 `422 CAPABILITY_UNSUPPORTED`，不能静默降级。该模式执行时定位失败，整个任务为 `failed`；v0.2 不提供部分成功结果。

`resolution_sec` 表示所用定位输出的有效时间粒度，不构成定位误差保证。0.16 秒粒度的模型不得通过插值宣称具有 0.001 秒精度。整段和定位可能来自不同输出头，因此允许整段标签与片段结果不一致；此时添加 `GLOBAL_LOCALIZATION_DISAGREE` 警告并分别展示，不得为消除差异而修改结果。

默认 `score_aggregation=timeline_mean` 时，`segments[*].spoof_score` 定义为合并后区间内、按有效覆盖时长加权的分数均值。允许合并规则包含短暂低分间隙，因此片段均值未必高于逐帧阈值。合并间隔、最短片段长度和平滑规则由 `postprocessing_version` 固定。

`score_aggregation=proposal` 表示候选片段原生分数，允许 `timeline=null`，不能补造片段之外的分数。片段仍须排序、不重叠且不越界。

具有真实时间格且请求 `include_timeline=true` 时，`timeline` 是适配器整理后的连续、不重叠时间格，每项包含 `start_sec`、`end_sec`、`spoof_score`，覆盖 `[0,duration_sec)`。重叠模型窗口先由适配器按固定规则融合；前端使用返回的时间边界，不自行推定帧率。未请求时 `timeline=null`。定位适配器需能取得带分数的候选片段或逐帧/窗口分数，不接受只有无分数起止区间的输出。后处理在原有时间粒度上进行，不得通过将全局分数复制到每一帧生成定位结果。

## 播放和音频预处理

成功结果中的 `audio.preview_url` 指向同一 API 下的 WAV 预览资源。建议按源音频解码时间轴输出 PCM WAV，保持起始静音、时长及时间对齐；有损编码的编码延迟由解码器处理并固定策略。模型使用的重采样波形与浏览器播放资源可以不同，但必须共享时间原点。

v0.2 预览文件较小，前端可完整获取为 Blob 后播放和跳转，不要求 HTTP Range。未成功时请求预览返回 `409 AUDIO_NOT_READY`，未知或过期分别为 `404/410`。跨域部署时由后端配置实际前端域名；部署采用登录态时，任务查询和预览沿用同一权限检查。当前草案没有约定可直接公网发布的身份方案。

## 错误约定

请求错误统一为 `{"error":{"code":"...","message":"...","retryable":false}}`。运行错误把同样的 error 对象放在任务体内。

| HTTP 或阶段 | 示例错误码 | 处理 |
| --- | --- | --- |
| `400` | `INVALID_REQUEST` | multipart 请求体无法解析 |
| `413` | `FILE_TOO_LARGE` | 超出上传大小 |
| `415` | `UNSUPPORTED_FORMAT` | 非支持的音频编码 |
| `422` | `INVALID_REQUEST`、`INVALID_AUDIO`、`DURATION_OUT_OF_RANGE`、`CHANNELS_UNSUPPORTED`、`CAPABILITY_UNSUPPORTED` | 请求参数或已知输入不符合要求 |
| `404` | `MODEL_NOT_FOUND`、`ANALYSIS_NOT_FOUND` | 资源不存在 |
| `409` | `AUDIO_NOT_READY` | 预览暂不可用 |
| `410` | `ANALYSIS_EXPIRED` | 数据已过期 |
| `429` | `QUEUE_FULL` | 暂时排队容量不足，可按提示重试 |
| `503` | `MODEL_UNAVAILABLE` | 模型未就绪，可重试 |
| `500` | `INTERNAL_ERROR` | 请求处理异常，返回通用信息和可排查日志 |
| 任务 `failed` | `INVALID_AUDIO`、`NO_VALID_AUDIO`、`DURATION_OUT_OF_RANGE`、`CHANNELS_UNSUPPORTED`、`MODEL_UNAVAILABLE`、`INFERENCE_TIMEOUT`、`INFERENCE_FAILED`、`INVALID_MODEL_OUTPUT` | 根据实际错误提示用户 |

创建前可判断的问题直接用 HTTP 错误返回；只有任务入队后解码才能发现的问题通过 `failed` 返回。建议单任务从创建起最长 10 分钟，超时进入 `failed/INFERENCE_TIMEOUT`；这是待部署确认的默认预算。空文件、全部采样值为零的纯静音返回 `NO_VALID_AUDIO`。近静音或非语音的识别阈值由所选模型预处理配置确定并随版本记录，尚未固定通用阈值；不能因识别不到语音就返回默认“真实”。所有未标明可重试的错误默认 `retryable=false`。

## 对接验收

1. 以三个 [结果示例](../examples/) 完成成功、有片段、无片段、不支持定位的展示，并以状态机覆盖等待和失败。
2. 使用一条带官方时间标注的局部伪造音频，从同一位置播放并标红，检查时间原点、片段边界及末尾没有超长。
3. 对真实模型核对分数方向、阈值、版本和有效时间粒度；人工构造的示例不得作为模型效果证据。
4. 检查超大文件、损坏文件、静音、时长越界、模型不支持定位的提示，以及任务过期后的停止轮询。

## 配套工程

本包已包含服务实现，启动方式见根目录 README。模型目录新增 `is_mock` 为本版与前一资料版本的主要差异。真实模型仍需提供源码、权重及推理配置；默认模拟模式不具备音频真实性检测能力。
