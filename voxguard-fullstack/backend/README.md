# 后端工程说明

当前实际 `.env` 已关闭模拟并使用真实 CFPRF 常驻适配器。请从外层项目根目录执行 `bash start.sh`，由启动器管理独立的后端和模型环境；停止使用 `bash start.sh --stop`。有效时长为 0.5–21.25 秒。具体配置与直接调用方式见 [CFPRF 常驻接入](../docs/cfprf-resident.md)。下文模拟默认值对应未加载 `.env` 的独立开发设置。

后端使用 FastAPI，提供模型目录、异步分析、结果查询和 WAV 预览，并在同一服务下提供前端静态文件。接口定义见 `../contracts/openapi.yaml`，版本为 v0.2。工程默认启用明确标记的模拟适配器；未随包交付经过训练的音频伪造检测模型。

## 启动

在**工程根目录**执行以下命令。建议使用 Python 3.11 或更新版本，并先安装 FFmpeg，确认 `ffmpeg` 与 `ffprobe` 均位于 PATH。

```bash
python -m pip install -r backend/requirements.txt
cp .env.example .env
python run.py
```

默认打开 `http://127.0.0.1:8000/`；接口说明位于 `/docs`，机器可读规范位于 `/openapi.json`。启动入口读取根目录 `.env`；已有进程环境变量优先。PyTorch 模型另需安装与模型代码及设备匹配的 PyTorch 和模型依赖，见 `../model_integration/README.md`。

采用一个服务进程和一个顺序推理执行器。不要启动多个 Uvicorn worker 共用数据目录；服务会使用独占文件锁阻止重复打开同一目录。当前工程适用于本地或受控环境的演示联调，不包含账户、身份验证或公网配额管理。

## 模型适配入口

环境变量 `VOXGUARD_ADAPTER_FACTORY=module:create_adapter` 指定无参数工厂函数。工厂返回一个适配器或适配器列表；创建时须完成配置与权重加载检查，只有 `describe().status=ready` 的模型进入目录。插件出错会使启动失败，不会退回模拟模型。真实模型部署建议设置 `ENABLE_MOCK=false`；禁用模拟且未指定工厂时目录为空，不能提交检测任务。

公共类型位于 `voxguard.adapters`：

```python
from voxguard.adapters import AudioInput, Prediction, AdapterError

def create_adapter():
    return CustomAdapter()

class CustomAdapter:
    def describe(self) -> dict:
        # 返回 Model 描述，包含 id/version/is_mock、能力、输入限制及冻结阈值。
        ...

    def predict(self, audio: AudioInput, mode: str, include_timeline: bool) -> Prediction:
        # 返回 Prediction(result=<AnalysisResult 字典>, analysis_sample_rate_hz=<实际采样率>)。
        ...
```

`AudioInput` 包含 `[channels, samples]` 的 CPU float32 波形、原始解码采样率 `sample_rate_hz`、有效时长 `duration_sec` 及任务标识 `audio_id`。不包含文件真值、样本标签或伪造区间。适配器负责真实的声道处理、重采样与模型输入转换。可复用 `voxguard.audio.resample_mono()` 执行等权单声道混合和 FFmpeg 重采样。

定位输出支持 `score_aggregation=proposal`：保留候选片段分数，允许 `timeline=null`，不计算虚构的逐帧均值。默认 `timeline_mean` 保持原有严格时间线校验。判定默认 `gte`；模型描述的 `decision_operator` 与结果的 `threshold_operator` 必须一致。

定位任务始终向适配器传入 `include_timeline=True`，以便服务校验实际提供的时间格；`timeline_mean` 模式校验完整时间格及片段的时间加权均值；HTTP 请求的 `include_timeline` 仅决定最终是否传输时间序列。服务严格校验有限分数、标签与冻结阈值、定位状态、真实时间边界、连续时间格及片段分数。未经校准的分数不限定为 0～1，不作为概率。

适配器可抛出 `AdapterError(code, message, retryable=False)`；其他异常统一记录到服务日志，任务进入失败状态。每次服务启动在独立子进程中发现和检查模型；每项任务也有独立 worker。当前 CFPRF worker 仅处理音频并请求常驻模型服务，不重复加载权重，HTTP 主进程不占用 CUDA 上下文。常驻模型随启动器一起退出；worker 超时终止不等于终止常驻模型进程。其他本地 PyTorch 适配器仍可采用每项任务加载权重的原有模式。

## 参数和数据保留

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `ENABLE_MOCK` | `true` | 是否注册显式模拟模型，返回结果带 `is_mock=true` |
| `VOXGUARD_ADAPTER_FACTORY` | 空 | 外部适配器工厂入口 |
| `VOXGUARD_DATA_DIR` | `data` | 相对于工程根目录的运行数据目录 |
| `QUEUE_CAPACITY` | `4` | 已运行和等待任务的总容量，满时返回 429 |
| `TASK_TIMEOUT_SEC` | `600` | 自任务创建开始计算的处理期限，包含等待和模型加载 |
| `MODEL_LOAD_TIMEOUT_SEC` | `120` | 启动时模型发现与加载检查的独立期限 |
| `RETENTION_SEC` | `86400` | 自创建起的结果及音频保留时间，须大于任务期限 |
| `CORS_ORIGINS` | 空 | 分离部署时允许的前端来源，使用逗号分隔的完整 origin；同源部署无需设置 |

运行目录包含任务记录 `record.json`、内部请求参数、原始上传文件、预览 WAV、内部输出和 `worker.log`；模型启动日志为 `startup.log`。这些文件不会通过静态路由公开。任务成功或失败后结果不再改变。服务重启时将尚未完成的任务标记为失败，保留已完成任务。

服务每 30 秒以及查询时清理到期的终态任务，只保留用于区分资源状态的最小标识记录；过期 ID 返回 410，未知 ID 返回 404。启动时清理异常中断遗留的临时上传文件。模拟数据与真实数据使用相同持久化及过期流程。

## 音频处理

- 仅允许实际容器为 WAV、FLAC 或 MP3，以 FFprobe/FFmpeg 识别结果为准；扩展名和客户端 MIME 不能替代解码校验。
- 文件上限为 20 MiB，支持一条单声道或双声道音频流；时长适用服务的 0.5～120 秒范围与模型时长范围的交集。
- 拒绝空文件、全部为零的采样、非有限采样、损坏音频和超出范围的时长；不将失败当成真实语音。
- FFprobe 和每次 FFmpeg 调用各限时 30 秒。解码采样率上限为 384 kHz。对压缩音频最多解码至允许时长再加 1 秒，以识别过长文件；超过允许时长的任务直接失败，不对截断音频出具结果。
- 推理输入保留原始解码 float32 幅度，不进行静默峰值归一化或削波。预览采用 PCM16 WAV 以兼容浏览器，对超出 PCM16 范围的播放幅度限幅，但不删静音、不裁剪或变速，保持同一时间原点和采样帧数。

## 验证

在工程根目录安装开发依赖并运行：

```bash
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests -q
```

后端含 27 项检查，覆盖上传、FFmpeg 解码、异步任务、检测与定位状态、OpenAPI 结构、预览时间轴、上传上限、异常音频、重采样、持久结果、过期清理、模拟关闭、非法模型输出、超时终止和有界队列。

已验证环境为 Linux、Python 3.13.5、FastAPI 0.135.1、Uvicorn 0.41.0、python-multipart 0.0.22、NumPy 2.1.3、PyYAML 6.0.2、HTTPX 0.28.1、pytest 8.3.4。其他操作系统或模型 CUDA 环境需按实际模型依赖验证。上述检查验证工程流程，不代表真实伪造检测准确率。
