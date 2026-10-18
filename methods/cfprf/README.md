# 音频伪造检测与定位模型

目前实现 CFPRF 单条音频推理，复用官方预训练 FDN + PRN，不需要训练。

已验证：HAD、PS、LAVDF 三套权重严格加载；公开 3.4 秒语音样例的 GPU 推理；HAD 的 CPU 推理及与 GPU 的输出对照；实际 PRN 权重的区间解码与官方评估器一致；3 项接口及解码单元测试通过。验证报告见 `results/validation.json`，测试日志见 `results/tests.txt`。

## 本机运行

以下命令在 `methods/cfprf` 目录执行，项目根目录位于其上两级。

```bash
bash scripts/cfprf.sh examples/audio/speech.wav --device cuda:0 --output results/speech.json
bash scripts/cfprf.sh your_audio.wav --variant HAD --device cpu --output results/result.json
```

可以使用 `--variant HAD|PS|LAVDF` 切换官方任务权重。默认 HAD，与官方评估脚本默认配置一致；并不意味着它在所有实际输入上性能最好。

## 命令参数

`audio` 是必填的音频文件路径，下面的参数均为可选项：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--variant` | `HAD` | 权重版本：`HAD`、`PS`（PartialSpoof）、`LAVDF` |
| `--device` | `auto` | 自动选择 CUDA 或 CPU，也可显式指定 `cpu`、`cuda:0` 等 |
| `--threads` | `4` | PyTorch CPU 线程数 |
| `--include-frames` | 不启用 | 额外输出帧级伪造分数 |
| `--output` | 标准输出 | JSON 保存路径；自动创建父目录，同名文件会覆盖 |
| `-h` / `--help` | — | 查看命令帮助 |

三种权重使用相同的模型结构与环境，每种会配套加载同版本的 FDN 和 PRN。下面的命令在**项目根目录**执行，使用 PS 权重测试 PartialSpoof 音频：

```bash
bash methods/cfprf/scripts/cfprf.sh voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --variant PS --device cuda:0 --include-frames \
  --output results—temp/CON_D_0000029_cfprf_ps.json
```

切换命令中的 `--variant` 只影响本次推理，不会修改网页常驻服务的权重配置。输出中的 `checkpoint_variant` 会记录实际使用的版本。

## 后端调用

后端需使用 `../../.envs/cfprf/bin/python` 所在环境调用以下接口。加载模型一次，然后逐条传入音频。

```python
from audio_forgery import CFPRFDetector

model = CFPRFDetector(variant="HAD", device="cuda:0")
result = model.predict(waveform, sample_rate=16000)
# 或者直接读取已经预处理好的音频文件：
result = model.predict_file("audio.wav")
```

输入约定：单声道、一维浮点 PCM 波形，通常幅度在 [-1, 1]，采样率固定 16000 Hz。公共解码、重采样及归一化由后端完成。本模块校验输入，不改变采样率、不进行幅度归一化。

输出为 JSON 可序列化字典：

```json
{
  "model": "CFPRF",
  "checkpoint_variant": "HAD",
  "sample_rate": 16000,
  "duration": 3.4,
  "is_fake": true,
  "label": "fake",
  "score": 0.91,
  "score_type": "max_refined_proposal_verification",
  "decision_rule": "at_least_one_refined_proposal_with_verification_gt_0.5",
  "frame_resolution": 0.02,
  "segments": [{"start": 0.8, "end": 1.6, "score": 0.91}]
}
```

以上数字仅说明格式，不是实际推理结果。`start`、`end` 单位为秒。区间分数来自 PRN 验证头，不是经过校准的真假概率。整段 `score` 是保留区间分数的最大值，没有保留区间时为 0；`is_fake` 根据是否有保留区间生成，是部署层的聚合规则。

`--include-frames` / `include_frames=True` 可额外返回 `frame_fake_scores`。第 i 帧对应约 i×0.02 秒，分数为 FDN class 0 的 softmax；官方标签 class 0 = fake，class 1 = real。编码器的卷积感受野导致尾部可能不足一帧，帧级输出不是逐采样点标注。

## 输入长度

单次输入支持 400–340000 个采样点，即 0.025–21.25 秒。上限沿用官方评估脚本的 340000 采样点限制，超长输入明确报错，不静默裁剪或跳过。更长音频需要后续制定窗口和跨窗口区间合并策略；这不属于当前原方法单条推理复现。

## 环境与源码适配

本机专用环境位于 `../../.envs/cfprf`，Python 3.10，复用已有 `py310` 环境的 PyTorch 2.5.1+cu121、torchvision、torchaudio；其他依赖安装到项目虚拟环境，未修改原 `py310` 环境。此环境需要保留基础 Python 环境，搬到其他机器时重新安装。

官方文档配置为 Python 3.8、PyTorch 1.8.1。本机针对 RTX 4090 使用较新的 PyTorch，因此是推理兼容适配，不能将单样例跑通等同于原论文指标复现。Fairseq 固定为作者指定提交 `a54021305d6b3c4c5959ac9395135f63202db8f1`。推理不需要的 Fairseq 原生训练扩展未编译。

重新配置环境：

```bash
CFPRF_BASE_PYTHON=/path/to/python3.10 bash scripts/setup_cfprf.sh
```

官方 CFPRF 源码保存在 `../../temp/CFPRF`。对 `models/FDN.py` 的兼容修改包括：允许显式传入 XLS-R 路径、允许选择设备、将位置编码注册为不写入 checkpoint 的 buffer，以便 `.to(device)` 正确移动它。数据加载器中已移除的 `torch._six.string_classes` 增加标准 Python 类型回退，使官方评估函数也能导入。FDN 和 PRN 的权重均严格加载；模型结构、分类标签、20 ms 帧分辨率、0.5 验证阈值、短区间处理及回归解码沿用原实现。

任务权重：`../../temp/weights/CFPRF`。共享编码器：`../../temp/weights/shared/xlsr2_300m.pt`。全部下载来源及哈希记录见 `../../temp/README.md`。

## 验证范围

```bash
../../.envs/cfprf/bin/python -m unittest discover -s tests -v
```

测试覆盖输入约定、空候选处理、真假类别方向、短区间、回归区间、验证阈值；适配器解码结果与官方 `Inference_CFPRF` 对照。实际模型推理验证记录保存到 `results/`。公开语音样例来源见 `examples/audio/SOURCE.md`。这验证部署链路，不代表已完成 HAD、PartialSpoof、LAV-DF 测试集上的论文指标复现。

三套真实权重的集成验证可重新执行：

```bash
../../.envs/cfprf/bin/python scripts/validate_cfprf.py --device cuda:0
```

其中强制粗候选区间仅用于确保真实 PRN 执行并对照官方解码，没有用于正常预测，不能当作伪造样本检测效果验证。CPU/GPU 样例的最大帧分数差约为 0.000278，两者均输出 `real`、空区间。
