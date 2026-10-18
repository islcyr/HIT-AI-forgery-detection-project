# TFCL 整段音频伪造检测

使用作者公开的 `TFCL_best_ckpt.pth`，无需训练。本目录提供单条音频 CLI 和 Python 接口，环境为项目根目录 `.envs/tfcl`。

## 直接运行

在**项目根目录**执行：

```bash
bash methods/tfcl/scripts/tfcl.sh voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --device cuda:0 --output methods/tfcl/results/result.json
```

CPU 使用 `--device cpu`。不指定 `--output` 时，标准输出为 JSON，依赖日志写入标准错误。结果父目录自动创建，同名文件会覆盖。

| 可选参数 | 默认值 | 说明 |
|---|---|---|
| `--device` | `auto` | 自动选择 GPU / CPU，也可指定 `cuda:0`、`cpu` |
| `--threads` | `4` | PyTorch CPU 线程数 |
| `--seed` | `1234` | 每次请求重置的随机种子 |
| `--threshold` | `0.5` | 伪造分数严格超过阈值才判 fake |
| `--output` | 标准输出 | 结果 JSON 路径 |
| `--checkpoint` | 本地 TFCL 任务权重 | 完整可信的任务 checkpoint |
| `--xlsr-checkpoint` | 本地共享 XLS-R | XLS-R 基础权重路径 |
| `-h` / `--help` | — | 查看帮助 |

TFCL 当前只有一套选用的任务权重，没有 CFPRF 那样的 `--variant` 选项。

## Python 接口

使用 `.envs/tfcl/bin/python`，在项目根目录运行：

```python
import sys
sys.path.insert(0, 'methods/tfcl')
from tfcl_runtime import TFCLDetector

model = TFCLDetector(device='cuda:0')  # 加载一次，重复调用
result = model.predict_file('your_audio.wav')
# 或由后端传入预处理后的数组：
# result = model.predict(waveform, sample_rate=16000)
```

输入一条 **16 kHz、单声道、浮点 PCM** 音频，数组是一维 float32。部署支持 400–340000 点，即 0.025–21.25 秒，这是接口资源限制，不是论文的通用时长上限。后端负责解码、重采样、声道转换和外部幅度归一化，本模块不做这些操作。

## 时长与模型行为

原版评估器每条只处理 **64600 点，约 4.0375 秒**：短音频重复补齐，长音频取开头。

本接口保留这个单窗口处理方式；为避免忽略后半段，超过64600点时切成**不重叠窗口**逐个检测，最后一个窗口重复补齐。整段分数取各窗口伪造分数的最大值。分窗与最大值聚合是部署层规则，尚未用目标验证集校准，不能直接当作论文的长音频评估结果。窗口越多，误报机会可能增加。

作者 `SSLModel.extract_feat()` 在评估时也显式执行 `self.model.train()`，开启 SSL 编码器的 dropout / layerdrop；本实现保留它，没有擅自改为 encoder.eval()。每次调用固定并恢复 PyTorch 和 NumPy RNG，以便相同环境、设备、种子下重复调用不随之前的请求变化；不同设备的随机路径和数值结果仍可能不同。推理不计算梯度，后端分类网络保持 eval 模式。

## 输出

返回 `model`、`checkpoint_variant`、`sample_rate`、`duration`、`is_fake`、`label`、`score`、`score_type`、`threshold`、`decision_rule` 等主要公共字段。

**这是整段检测方法，不提供伪造定位：** `supports_localization=false`、`segments=null`。`windows` 只是计算窗口及其分数，不是预测伪造片段。

- 第 0 类是 spoof / fake，第 1 类是 bonafide / real。
- 每窗口分数为 `softmax(logits)[0]`，整段 `score` 是最大窗口分数，未做概率校准。
- `windows` 记录实际输入的 `start`、`end`、`score`、原始 `logits` 和 `raw_bonafide_score`。后者对应官方评估脚本导出的第 1 类原始 logit，越大越偏真实。
- 任意窗口分数严格大于阈值则整段判 fake；0.5 是部署默认规则，官方 EER 阈值并不是固定的0.5。
- 输出还记录 `window_seconds=4.0375`、`ssl_mode=train_as_upstream` 和 `seed`。

真实样例结果保存在 `results/`。本方法未自动加入五版本定位路由，也未替换网页默认模型。

## 权重、环境与来源

两份权重均已存在，无需补下载：

- `temp/weights/TFCL/TFCL_best_ckpt.pth`：作者提供的单分支推理权重，674 项状态严格加载；仅用于训练的一致性分支不参与推理。
- `temp/weights/shared/xlsr2_300m.pt`：共享 XLS-R 300M 编码器。

任务权重使用 `weights_only=True` 读取；编码器沿用 Fairseq 的原加载方式。为了严格检查单分支权重，初始化后移除没有参与单分支 forward 的 `bidirectional_attn_T` 和 `feature_proj_d` 模块，再执行 `strict=True`；不会容忍缺失推理参数。

官方源码：[JunXue-tech/TFCL](https://github.com/JunXue-tech/TFCL)，本地为 `temp/TFCL`，固定提交 `d6d67dd7e3f411151fae75cc2578d62066d2a413`。`tfcl_runtime/upstream_model.py` 复制原版 `code/model.py`，只将写死的 XLS-R 路径改为显式构造参数，保留网络计算。上游许可证见 `UPSTREAM_LICENSE`。

环境使用 Python 3.10、PyTorch / torchaudio 2.5.1+cu121、NumPy 1.23.5。Fairseq 使用本地固定提交 `a54021305d6b3c4c5959ac9395135f63202db8f1`，不构建训练原生扩展。虚拟环境继承本机 py310 基础 PyTorch，其余依赖单独安装。

```bash
TFCL_BASE_PYTHON="$(conda run -n py310 python -c 'import sys; print(sys.executable)')" \
  bash methods/tfcl/scripts/setup_tfcl.sh
```

## 验证

```bash
.envs/tfcl/bin/python -m unittest discover -s methods/tfcl/tests -v
.envs/tfcl/bin/python methods/tfcl/scripts/validate_tfcl.py --device cuda:0
.envs/tfcl/bin/python methods/tfcl/scripts/validate_tfcl.py --device cpu
```

验证涵盖严格权重加载、真实/全伪造/部分伪造音频的实际调用、同实例固定种子重复推理、输入最短/最长长度和分窗末尾覆盖。报告分别位于 `results/validation_cuda.json`、`results/validation_cpu.json`。单元测试验证原版补齐方式、输入约束、随机状态恢复、末窗参与判定以及无定位输出。此处验证部署流程，没有复现论文完整 EER / AUC 指标。
