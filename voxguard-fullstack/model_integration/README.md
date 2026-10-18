# 真实模型接入与验证

本目录提供可配置的 PyTorch 权重加载桥接、模型入口模板、自定义适配器模板和单音频检查工具。真实检测需要接收方提供可执行的模型源码、与其匹配的权重和明确的分数及时间语义；仅有任意权重文件不能自动推断模型结构。

**模型的原始输出可能需要适配器转换。** 可先按[完整操作示例](../docs/model-adapter-walkthrough.md)运行“元组输出 → 标准分数 → 前端展示”流程。示例有可执行代码与测试权重，始终标记为模拟，不具有伪造检测能力。

## 文件用途

| 文件 | 用途 |
| --- | --- |
| `torch_adapter.py` | 加载源码及权重，处理声道和采样率，执行实际推理，生成统一结果 |
| `output_adapter_example.py` | 将元组中的 logits 转换为通用桥接所需的整段及逐格分数 |
| `make_output_adapter_example.py` | 生成可执行适配示例的源码、测试权重和配置 |
| `OUTPUT_ADAPTER_EXAMPLE.md` | 示例输出、包装层与权重匹配的具体说明 |
| `model-config.example.json` | 配置模板；包含占位值，不是已配置模型 |
| `model_entry.template.py` | 按模型源码填写 `create_model` 的入口模板 |
| `custom_adapter.template.py` | 不适用通用桥接时，实现统一模型接口的模板 |
| `check_model.py` | 对单条音频执行真实推理，并调用后端结果校验 |
| `requirements-model.txt` | 通用桥接附加依赖；还需安装模型自身依赖 |
| `plumbing_fixture.py`、`make_fixture.py` | 生成并运行权重加载链路测试夹具，不具备伪造检测能力 |

## 通用桥接支持范围

模型工厂形如 `module:create_model`，返回一个 `torch.nn.Module`；一次 `forward` 接收 `[1,T]` 或 `[1,C,T]` 浮点波形。整段输出必须是一个有限分数。定位输出必须能明确解释为从原始音频第 0 秒开始、相邻且无重叠的时间格；支持 `[T]` 或 `[1,T]` 分数数组，最后一格可短于固定步长。模型需要对传入波形的完整长度输出 `ceil(duration_sec / resolution_sec)` 个时间格，不能漏掉首尾。

需特征提取、额外长度参数、类别 logits 转换、重叠窗口融合、固定长度裁剪或分块推理的模型，应在模型源码入口中增加明确的包装，或实现 `custom_adapter.template.py`。通用桥接不会自动将整段分数复制成定位结果，也不会猜测卷积 padding、时间偏移或类别轴。对不支持定位的模型，将 `localization` 和 `thresholds.localization` 设为 `null`，只能请求整段检测。

## 接入步骤

1. 将模型源码、权重和配置存入独立目录，例如项目根目录的 `models/custom/`，其下包含 `source/`、`weights/model.pt` 和 `config.json`。
2. 将配置模板复制为 `config.json`，填写模型 ID、名称、发布版本、源码版本、工厂函数、构造参数和运行设备。`source_dir` 和 `weights` 都相对于配置文件所在目录解析，整体移动资料包不会破坏相对路径。
3. 根据源码填写 `source/model_entry.py` 的 `create_model(**kwargs)`；工厂创建网络结构即可，统一桥接负责加载权重。必须提供完整且匹配的 `state_dict`，不接受忽略缺失参数后继续运行。
4. 填写检查点 SHA-256，可使用 `sha256sum weights/model.pt` 获取；`state_dict_key` 为嵌套字典路径，例如 `state_dict`，若权重文件直接保存参数字典则为 `null`。桥接使用 `torch.load(..., map_location="cpu", weights_only=True)` 和 `load_state_dict(..., strict=True)`。缺少文件、哈希不符、结构不符或设备不可用时，服务启动失败，不登记为就绪模型。
5. 明确输入采样率、声道策略和形状。`channels=mean` 对多声道取均值；`preserve` 保留声道且要求 `layout=BCT`。不同采样率使用多相滤波实际重采样，不修改文件头冒充重采样，不删除前导静音。
6. 填写输出字段路径，如 `scores.global`、`scores.frames`；根输出就是单个分数时 `global_key=null`。填写分数方向；若原模型越大越真实，设 `higher_is_bonafide`，桥接将整段及逐格分数乘以 -1。**配置中的阈值必须已转换为“越大越伪造”的统一方向**，并来自该模型确定的开发集设置；示例数值只是字段示意，不能直接当作已验证阈值。
7. 填写定位步长 `stride_samples`，单位为模型采样率下的采样点，`start_offset_samples` 必须为 0。桥接将高于或等于定位阈值的相邻时间格合并，片段分数按有效时长加权；不平滑、不跨低分间隙合并。若需要其他后处理，编写自定义适配器并更新后处理版本。

版本同时记录配置指纹及权重哈希，配置或权重变化会改变接口返回的 `model.version`。源码修改时必须更新 `source_revision`，阈值及后处理修改时必须更新对应版本字段。模型加载成功仅表明结构和环境可加载；投入联调前仍须执行以下单音频检查。

## 检查并启动

以下命令均从资料包根目录执行。先按根目录 README 安装后端依赖，并在相同 Python 环境安装与模型和 CPU/CUDA 环境匹配的 PyTorch、`requirements-model.txt` 以及模型自身依赖。

```bash
python -m pip install -r model_integration/requirements-model.txt
python -m model_integration.check_model --config models/custom/config.json --audio samples/audio/CON_D_0000000.wav --output model-check.json
```

检查工具实际加载权重、执行单条音频推理并验证输出，失败返回非零退出码。输出中的 `is_mock`、版本、分数、阈值、时间格和片段应逐项核对。格式检查不等于检测准确率评估。

启动时设置以下环境变量，使用根目录 README 中的服务启动命令：

```bash
export ENABLE_MOCK=false
export VOXGUARD_ADAPTER_FACTORY=model_integration.torch_adapter:create_adapter
export VOXGUARD_MODEL_CONFIG=models/custom/config.json
```

服务启动后，前端模型列表将展示配置中的模型。上传音频将经过真实模型推理，页面使用后端返回的整段分数及时间片段。不要把 `is_mock` 改为 false 来伪装测试夹具；真实模型须同时提供真实模型源码、权重和经核对的配置。

## 验证权重加载链路

```bash
python -m model_integration.make_fixture --output /tmp/voxguard-plumbing-fixture
python -m model_integration.check_model --config /tmp/voxguard-plumbing-fixture/config.json --audio samples/audio/CON_D_0000000.wav
```

生成目录包含自足的源码、约数 KB 权重和带实际哈希的配置；所有结果始终为 `is_mock=true`，模型 ID 为 `example-plumbing-fixture`。该夹具将波形能量通过简单参数转换为分数，只用于验证“源码 → 权重 → 波形 → 推理 → 时间输出”的工程链路，不能用其结果判断真假，也不能据此报告检测性能。

## 其他模型框架

复制并实现 `custom_adapter.template.py`，将 `VOXGUARD_ADAPTER_FACTORY` 改为该模块的 `create_adapter`。后端传入 `AudioInput`，返回 `Prediction`；定位请求会要求完整时间格用于服务端一致性校验，然后按用户请求决定是否向前端公开该序列。模型执行错误应抛出 `AdapterError`，不得默认返回“真实”或回退至随机、固定、标签派生的分数。
