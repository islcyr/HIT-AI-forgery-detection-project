# 可执行的模型输出适配示例

模型源码的 `forward()` 输出不一定与服务要求一致。现有通用 `TorchAdapter` 接收字典中的一个整段分数及逐格分数；如果模型返回元组、类别 logits 或额外特征，需要先增加输出包装层。前端仍使用现有 API，无需修改。

本示例实际创建并加载一个小型 PyTorch 网络，演示完整转换链路。**网络未训练，基于波形能量构造测试输出，始终标为模拟联调，不具备真假检测能力。** 示例不使用样本标签生成预测，不加载外部检测权重，不运行 GPU。

## 文件与转换关系

| 文件或组件 | 职责 |
| --- | --- |
| `output_adapter_example.py` 中的 `MockTupleNetwork` | 返回 `(frame_logits, utterance_logits, auxiliary)`，分别为 `[B,F]`、`[B,1]` 和辅助信息 |
| 同文件中的 `TupleOutputAdapter` | 检查输入和两个预测张量的形状及有限值，转换分数方向，返回 `global_score` 和 `frame_scores` |
| `make_output_adapter_example.py` | 生成自足的模型源码、完整包装层权重及包含真实 SHA-256 的配置 |
| 现有 `torch_adapter.py` | 严格加载权重、预处理音频、执行包装层、按原始时间轴生成片段及统一结果 |

本示例明确约定原始 logits 越大越倾向真实，因此包装层对两个预测头都执行 `1 - sigmoid(logit)`。例如，logit 为 `-2`、`0`、`2` 时，对应伪造方向分数约为 `0.880797`、`0.5`、`0.119203`。这些数值未经概率校准，不能作为真实性概率。配置使用 `higher_is_spoof`，避免重复翻转方向。

整段分数来自独立的示例整段输出，不能用辅助特征或最大帧分数替代。辅助信息不参与结果。用于形状检查的 `F` 是 `ceil(N / 1600)`，其中 `N` 是 16 kHz 模型输入的采样点数；不接受丢失最后一格的输出。

## 生成并检查

以下命令从工程根目录执行，Python 环境需安装后端依赖和 `model_integration/requirements-model.txt`。CPU 即可运行。

```bash
python -m model_integration.make_output_adapter_example --output models/output-adapter-example
python -m model_integration.check_model --config models/output-adapter-example/config.json --audio samples/audio/CON_D_0000000.wav --output output-adapter-check.json
```

生成器不会覆盖已有的源码、配置或权重；重新生成时选择新的输出目录。生成目录包含：

```text
models/output-adapter-example/
├── config.json
├── output_adapter_example.pt
└── source/
    └── vg_output_adapter_example.py
```

固定模型 ID 为 `example-output-adapter`，版本前缀为 `example-output-adapter-v1`，实际接口版本还包含配置指纹和权重哈希。生成配置中的 `is_mock` 始终为 `true`，设备固定为 `cpu`，工厂入口为 `vg_output_adapter_example:create_model`。配置中的源码和权重使用相对路径，可整体移动目录。

接入服务时沿用通用桥接入口：

```bash
export ENABLE_MOCK=false
export VOXGUARD_ADAPTER_FACTORY=model_integration.torch_adapter:create_adapter
export VOXGUARD_MODEL_CONFIG=models/output-adapter-example/config.json
```

然后按工程根目录 README 启动后端。`ENABLE_MOCK=false` 关闭内置演示模型，不会将这个输出适配示例变为真实模型；其模型目录和任务结果仍保留 `is_mock=true`。

## 时间与阈值语义

16 kHz 输入每 1600 个样本形成一格，即 100 ms。网络对最后不足 100 ms 的真实采样点独立计算，不丢弃末尾、不复制前一格分数。内部补零只用于整形，均值除以实际有效样本数。例如 0.65 秒音频输出 7 格，最后一格为 `[0.6,0.65)` 秒，全部格子覆盖原始时长。

示例整段与定位阈值均为 `0.5`，使用现有通用桥接的“大于或等于”规则。这仅为流程示例，未做阈值标定。相邻高分格子的合并与时间加权片段分数由现有桥接完成。

实际模型只有在输入形状、类别方向、独立整段输出、逐格覆盖规则均相同时，才能复用此转换。类别轴、重叠窗口、起始偏移、遗漏尾段或无整段输出的模型，需要按实际结构编写相应适配器；不能仅修改字段名称或机械应用 `1 - sigmoid`。包装层结构改变时，还需要与其参数名称严格对应的权重；本例保存的参数带 `network.` 前缀，并通过 `strict=True` 加载。

## 测试

```bash
python -m pytest tests/model_integration/test_output_adapter_example.py -q
```

测试覆盖分数方向、独立整段输出、错误形状、非有限值、最后不足一格的时间覆盖、完整权重加载及不覆盖已有文件。通过这些检查仅表明输出适配链路符合接口要求，不代表检测准确率通过验证。
