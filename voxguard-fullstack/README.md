# VoxGuard 音频伪造检测与定位工程

**本机已接入真实 CFPRF 常驻服务。** 请在外层项目目录执行 `bash start.sh`，模型和后端将使用两个独立环境启动。实际配置 `.env` 已禁用模拟；有效时长为 0.5–21.25 秒。直接调用和验证记录见 [CFPRF 常驻接入说明](docs/cfprf-resident.md)。下文模拟模式说明对应原始 `.env.example`，不是当前实际配置。

本工程包含前端、Python 后端、模型接入组件、接口规范及 30 条带详细标签的音频。前端通过后端执行上传、任务查询、分数展示、时间片段标记、定位试听和报告导出。

默认运行明确标识的模拟联调模型，可以验证完整交互流程。包内不含经过训练的音频伪造检测模型或相应权重。接入真实模型后，前端及后端业务代码无需修改；模型源码、权重和配置须符合下述统一接入约定。

**模型输出可能需要适配器。** 仅取得模型代码和参数权重，不一定能直接连接页面。模型返回元组、多尺度张量、类别 logits 或特殊时间窗口时，需要适配器将其转换为统一分数和时间片段。本包提供可运行的[输出适配器示例及全流程操作说明](docs/model-adapter-walkthrough.md)；示例为工程测试模型，不具有真实伪造检测能力。前端保持上一交付版本不变。

## 1. 启动完整工程

需要 Python 3.11 或更高版本，以及已加入 PATH 的 FFmpeg、FFprobe。推荐先创建独立环境；已有模型环境也可以安装后端依赖后使用。前端由同一服务提供，不需要 Node.js、npm 或额外构建。

在解压后的工程根目录执行：

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 使用：.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python run.py
```

浏览器访问 <http://127.0.0.1:8000>。选择一个模型，从 `samples/audio/` 上传 WAV 文件，再点击开始检测。默认目录包含“模拟定位模型”和“模拟整段模型”，所有模拟结果都明确标识；模拟分数和片段不能用于评价音频真实性。

服务与网页使用同一地址，默认不需要跨域设置。接口文档位于 `/docs`。使用 `python run.py --host 0.0.0.0 --port 8000` 可以监听其他网卡，适用于有访问控制的内部联调环境。本工程未包含账号登录或公网部署方案。

CFPRF 命令行输出及后续不同格式模型的接入，见 [命令模型接入说明](docs/command-model-integration.md)。支持只有定位片段、没有逐帧时间线的模型。

## 2. 接入模型源码和权重

先阅读 [模型接入说明](model_integration/README.md)。符合配置式入口要求的 PyTorch 模型只需提供模型创建函数、匹配的 state_dict 权重及接入配置。配置包含采样率、输入形状、输出字段、分数方向、阈值和定位时间规则。

将根目录 `.env.example` 复制为 `.env`，按模型接入说明设置：

```dotenv
ENABLE_MOCK=false
VOXGUARD_ADAPTER_FACTORY=model_integration.torch_adapter:create_adapter
VOXGUARD_MODEL_CONFIG=models/custom/config.json
```

安装模型自身依赖，运行模型检查工具，通过后执行 `python run.py`。模型加载失败会明确报错，真实模式不会回退到模拟结果。

具有特殊特征提取、多个输入、重叠窗口或特殊输出结构的模型，应按随包模板实现一个适配器。仅有权重而缺少网络结构、推理预处理、标签含义或帧时间定义时，不能自动恢复可靠的推理行为。接入工作集中在模型目录及配置，前端和 HTTP 接口保持一致。

只有整段分类输出的模型支持整段检测；时间片段定位需要模型提供逐帧或逐窗口分数及明确时间映射。工程不会把整体分数复制成定位结果。

## 3. 文件目录与用途

| 文件或目录 | 内容与使用方式 |
| --- | --- |
| `run.py` | 加载根目录 `.env`，启动后端及同源前端 |
| `.env.example` | 模拟模式、模型入口、存储、队列及跨域配置示例 |
| `frontend/` | 修正后的原生 HTML、CSS、JavaScript；无需构建 |
| `backend/` | 上传、解码、任务调度、模型调用、结果校验、预览及后端测试 |
| `model_integration/` | PyTorch 权重加载、配置式适配器、自定义适配模板及检查工具 |
| `model_integration/output_adapter_example.py` | 可执行的元组输出适配示例：检查输出形状并转换分数方向 |
| `model_integration/make_output_adapter_example.py` | 生成上述示例的源码、测试权重与配置 |
| `docs/model-adapter-walkthrough.md` | 从生成示例、启动服务到替换真实模型的完整操作步骤 |
| `contracts/openapi.yaml` | API v0.2 字段规范；URL 仍为 `/api/v1` |
| `contracts/*schema.json` | 样本清单和详细标签的格式定义 |
| `examples/` | 合成 API 返回示例；不代表模型预测 |
| `samples/audio/` | 30 条 WAV，真实、整段伪造、局部伪造各 10 条 |
| `samples/labels/` | 汇总 CSV、片段 CSV、逐条 JSON、原始标签和协议摘录 |
| `samples/manifest.jsonl` | 按行读取的音频清单，路径相对于工程根目录 |
| `samples/source/` | 数据来源、原始许可说明及整理记录 |
| `docs/` | 接口语义、交付说明、数据标签说明及验证结果 |
| `tests/` | 模型接入及完整流程验证；测试模型不具有检测能力 |
| `tools/` | 样本与标签校验、HTTP 样本联调、可选浏览器验收和包完整性检查 |
| `THIRD_PARTY_NOTICES.md` | 音频来源、署名与再分发说明 |
| `SHA256SUMS.txt` | 包内文件校验值，不含该文件自身 |

## 4. 查看音频及标签

从 [音频目录说明](samples/README.md) 或 [标签汇总](samples/labels/summary.csv) 选择文件。逐条详细标注位于 `samples/labels/detailed/`，包含来源、真实类别、伪造区间及未标注区域等信息。标签语义详见 [标签说明](docs/label-guide.md)。

音频标签属于数据集真值，服务不会将这些标签传入模型或用作检测返回值。模型结果与真值对比应在独立评估中进行；30 条联调样本不足以证明模型准确率。

## 5. 行为与运行边界

- 上传格式为 WAV、FLAC、MP3，大小不超过 20 MiB，时长 0.5～120 秒，单声道或双声道；模型可以规定更严格的时长范围。
- 分数越大越倾向伪造，但不是概率。界面使用模型返回的判定和阈值，不显示未经校准的百分比。
- 定位采用原始音频秒数；“未检出片段”“未请求定位”“模型不支持定位”分别展示。
- 点击“停止等待”只停止浏览器查询，服务端已创建的任务仍可继续运行。
- 默认单服务进程、单推理任务串行执行，任务超时 600 秒，保留 24 小时。服务重启时未完成任务被明确标记失败，已完成结果保留到过期。
- 每个任务在独立进程加载适配器和权重，以便超时终止。此实现适合基础演示；大型模型的重复加载成本需要纳入耗时，不宣称具备常驻模型服务的吞吐性能。

## 6. 验证与交付范围

[交付验证记录](docs/validation.md) 列出实际执行的检查与范围。开发测试依赖、命令及模型接入验证详见相应目录 README。模型权重不随工程打包，依赖环境和 FFmpeg 需要在接收方机器上安装。

API 变更和前后端修正详见 [工程交付说明](docs/engineering-handoff.md)。
