# 模型输出适配与完整使用流程

## 交付范围

工程包含原有前端、后端、30 条带详细标签的音频、通用权重加载组件以及一个可运行的输出适配示例。前端与上一版交付内容一致，不需要修改 HTML、CSS 或 JavaScript。

**不同模型的输出可能需要适配器。模型源码与参数权重是必要条件，模型输入和输出不符合接口约定时，还需要完成适配。** 包内的可执行示例演示转换过程，不能替代训练后的检测模型。

数据流为：上传音频 → 后端解码 → 模型预处理与推理 → 输出适配 → 标准结果校验 → 前端展示分数和伪造时间片段。前端始终调用同一套 API，不读取模型的原始张量。

## 1. 准备环境

进入解压后的 `voxguard-fullstack` 目录。使用 Python 3.11 或以上版本，并准备可执行的 FFmpeg、FFprobe。建议使用独立环境；已有模型环境可以直接安装缺少的依赖。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m pip install -r model_integration/requirements-model.txt
```

Windows PowerShell 中，环境激活命令为 `.venv\Scripts\Activate.ps1`。PyTorch 应与所用 CPU 或 CUDA 环境匹配；本示例可以使用 CPU。

## 2. 生成可运行的适配器示例

```bash
python -m model_integration.make_output_adapter_example --output models/output-adapter-example
```

生成器会输出模型源码、测试权重和 `config.json`。示例底层模型返回 `(frame_logits, utterance_logits, auxiliary)`，输出包装器将其转换为 `{"frame_scores": ..., "global_score": ...}`，供现有通用适配器读取。

示例约定 logits 越大越倾向真实，因此转换为 `1 - sigmoid(logits)`，统一成越大越倾向伪造的分数。该转换仅适用于已经确认这一类别方向的模型，不能直接套用到未知模型。示例具有明确的 100 毫秒时间格，并处理最后不足一格的真实音频；不会将整段分数复制为定位分数。

生成的模型始终标记为 `is_mock=true`，页面明确显示模拟说明。模型参数实际参与计算，但模型没有接受伪造检测训练，分数和片段不构成真实性判断。

## 3. 单条音频检查

```bash
python -m model_integration.check_model \
  --config models/output-adapter-example/config.json \
  --audio samples/audio/CON_D_0000000.wav \
  --output output-adapter-check.json
```

检查工具实际加载源码及权重、执行音频推理、检查数值与时间范围，并保存结果。失败时返回非零退出状态，不能用空片段或零分替代错误。

## 4. 启动前后端

将根目录 `.env.example` 复制为 `.env`，设置以下三项，其余默认配置可以保留：

```dotenv
ENABLE_MOCK=false
VOXGUARD_ADAPTER_FACTORY=model_integration.torch_adapter:create_adapter
VOXGUARD_MODEL_CONFIG=models/output-adapter-example/config.json
```

`ENABLE_MOCK=false` 关闭内置模拟目录；示例自己的 `is_mock=true` 标识仍然保留，不会因此变成真实检测模型。外部环境变量优先于 `.env`；若此前设置过上述变量，需更新或清除冲突值。

```bash
python run.py
```

浏览器打开 <http://127.0.0.1:8000>。操作顺序为：选择输出适配示例 → 上传 `samples/audio/` 中的 WAV → 选择“检测与定位” → 开始检测 → 查看分数、时间片段与播放位置 → 导出 JSON 报告。

接口文档地址为 <http://127.0.0.1:8000/docs>。启动在服务器上时，可通过端口转发访问；需要局域网访问时按根 README 设置监听地址。模拟模式的结果仅用于验收上传、模型调用和页面展示链路。

## 5. 替换为真实模型

1. 准备匹配的模型源码和权重，安装模型自身依赖。确认采样率、声道处理、输入形状及允许时长。
2. 对照 `output_adapter_example.py` 替换网络创建函数和输出转换逻辑。确认元组各项含义、类别方向、是否有原生整段输出、定位帧的时间步长及首尾覆盖策略。
3. 保证包装后的参数键名与权重严格匹配。不能忽略缺失参数继续推理；包装器新增网络层级时，参数名可能增加前缀，需使用匹配的入口或明确的权重转换。
4. 按真实模型填写 `config.json`：源码入口、参数、权重 SHA-256、版本、采样率、整段与定位输出字段、阈值、时间格规则、设备及 `is_mock=false`。阈值不能沿用示例值冒充已验证值。
5. 先运行 `check_model`，用模型提供方的参考结果核对分数和时间片段；通过后将 `.env` 的配置路径指向真实模型，再启动服务。

输出已经是合约要求的字典时，可以直接配置字段路径。模型需要多输入、复杂重叠窗口或特殊后处理时，应实现 `custom_adapter.template.py` 中的完整适配器，并将工厂入口指向该适配器。

当前共享接口要求定位时间序列连续覆盖原始有效音频。原模型缺少尾帧时，必须先确定合理且明确的时间映射或修改接口约定；不得拉伸帧、复制末帧，或将未预测尾段标成真实来掩盖缺失。仅有整段输出的模型只能开启整段检测，不能据此生成精确定位。

## 6. 交付给模型提供方的最少要求

| 项目 | 必需内容 |
| --- | --- |
| 运行资源 | 网络源码、依赖、匹配权重、额外预训练组件 |
| 输入 | 采样率、声道、幅度处理、形状、时长限制 |
| 原始输出 | 各张量含义、形状、类别顺序、分数方向 |
| 时间映射 | 步长、首帧偏移、重叠融合、尾段处理 |
| 决策 | 固定阈值、整段分数是否派生、片段后处理 |
| 核对依据 | 权重校验值、版本及至少一条参考推理结果 |

30 条带标签音频用于联调与结果核对，详细标签见 `samples/labels/`。模型推理不读取真值标签；此示例不提供模型检测准确率结论。
