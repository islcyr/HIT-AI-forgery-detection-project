# Nes2Net-X 单条音频检测

复用作者公开的 **Nes2Net-X DF-best 单 checkpoint**，不训练。它是整段检测模型，不提供伪造片段定位。独立环境位于项目根目录 `.envs/nes2net`。

## 运行

在项目根目录执行：

```bash
bash methods/nes2net/scripts/nes2net.sh voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --device cuda:0 --output methods/nes2net/results/result.json
```

CPU 改为 `--device cpu`。不指定 `--output` 时标准输出为 JSON，日志写到标准错误。输出父目录自动创建，同名文件会覆盖。

**默认只输出原始分数，不使用未经验证的0.5阈值生成真假结论。** 作者的单条示例输出 `model(x)[:, 1]`，这是原始真实 logit，越高越偏真实。我们尚未为该权重选择部署阈值，因此 `threshold=null`、`is_fake=null`、`label=unclassified`。

后续用独立校准数据确定阈值后，可显式传入 `--threshold <原始真实logit阈值>`，判断规则为：`score < threshold` 判 fake，否则 real。TFCL 的阈值不能移用到这里，也不能用分数是否大于0.5或0来代替校准。

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--device` | `auto` | 自动选择 GPU / CPU，或指定 `cuda:0`、`cpu` |
| `--threads` | `4` | PyTorch CPU 线程数 |
| `--test-mode` | `full` | `full`处理完整音频，`4s`取前4秒或重复补齐 |
| `--threshold` | 不设置 | 原始真实logit的校准阈值，低于它判伪造 |
| `--checkpoint` | 本地 DF-best 单模型权重 | 指定完整可信 checkpoint，严格加载 |
| `--xlsr-checkpoint` | 本地共享 XLS-R | 基础编码器路径 |
| `--output` | 标准输出 | 结果 JSON 路径 |
| `-h` / `--help` | — | 查看帮助 |

当前只有这一套选用权重，不提供虚构的 `--variant`。作者还发布了其他权重及 checkpoint 平均版本，但本机没有下载，也没有在此复现。

## Python 接口与输入

使用 `.envs/nes2net/bin/python`：

```python
import sys
sys.path.insert(0, 'methods/nes2net')
from nes2net_runtime import Nes2NetDetector

model = Nes2NetDetector(device='cuda:0', test_mode='full')
result = model.predict_file('your_audio.wav')
# 或由后端传入预处理后的数组：
# result = model.predict(waveform, sample_rate=16000)
```

输入为一条16 kHz、单声道、一维浮点 PCM 音频，通常为 float32。部署限制为400–340000采样点，即0.025–21.25秒；这是本接口的资源限制，不是原模型的固定长度限制。后端负责解码、重采样、声道转换和外部幅度预处理。本模块只校验，不改变这些属性。

- `full`：默认与作者 `easy_inference_demo.py` 的默认模式一致，完整音频一次前向推理，无额外分窗和最大值聚合。
- `4s`：沿用作者单条示例，将短音频重复补齐到64000点，长音频只取前64000点。输出明确记录实际覆盖范围。
- 注意作者批量数据加载器中的 `4sec` 使用64600点，与单条示例的64000点不同；本接口 `4s` 明确对齐单条示例，不将两者混称为完全相同的评估配置。

## 输出

主要字段：

```json
{
  "model": "Nes2Net-X",
  "checkpoint_variant": "DF-best-single",
  "sample_rate": 16000,
  "duration": 3.1,
  "score": -1.2,
  "score_type": "raw_bonafide_logit",
  "score_direction": "higher_is_real",
  "is_fake": null,
  "label": "unclassified",
  "threshold": null,
  "supports_localization": false,
  "segments": null,
  "test_mode": "full",
  "processed_samples": 49600,
  "evaluated_end_seconds": 3.1
}
```

上面数值仅展示格式，不是实测结果。输出还保留两类原始 `logits`、辅助 `fake_softmax_score` 和 `decision_rule`。第0类伪造、第1类真实；辅助softmax不用于自动判定，也不是校准概率。`score` 无0–1范围限制，不能直接显示为百分比或与定位模型分数平均。

## 权重、源码与环境

- 任务权重：`temp/weights/Nes2Net/Nes2Net-X_DF_best.pth`，README中 DF EER 1.65% 对应的单checkpoint，989项状态全部 `strict=True` 加载。
- 编码器：`temp/weights/shared/xlsr2_300m.pt`，复用共享文件。任务权重以 `weights_only=True` 读取，编码器按原版 Fairseq 流程加载。
- 官方源码：[Liu-Tianchi/Nes2Net_ASVspoof_ITW](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW)，本地 `temp/Nes2Net` 固定提交 `c9fe1177f44040f002cb40def928332d2e261f1e`。
- `nes2net_runtime/upstream_model.py` 复制作者 `model_scripts/wav2vec2_Nes2Net_X.py`，只将编码器的写死路径改为显式参数。网络配置为 Nes_ratio=[8,8]、SE_ratio=[1]、dilation=2、mean pooling、2类输出，与作者默认设置一致。
- 先将整个模型移动到目标设备，再执行 eval，推理时保持SSL编码器和分类后端为eval模式，不触发原版“输入设备或类型改变时切换train”的分支。
- 上游README声明Apache-2.0许可；原模型文件中的作者署名保留。

环境：Python3.10，PyTorch / torchaudio2.5.1+cu121，NumPy1.23.5，Fairseq固定为本地 `a54021305d6b3c4c5959ac9395135f63202db8f1`。本机虚拟环境继承py310基础PyTorch，其余推理依赖独立安装。不需要 s3prl、Transformers、Lightning 或训练增强依赖。

```bash
NES2NET_BASE_PYTHON="$(conda run -n py310 python -c 'import sys; print(sys.executable)')" \
  bash methods/nes2net/scripts/setup_nes2net.sh
```

## 验证

```bash
.envs/nes2net/bin/python -m unittest discover -s methods/nes2net/tests -v
.envs/nes2net/bin/python methods/nes2net/scripts/validate_nes2net.py --device cuda:0
.envs/nes2net/bin/python methods/nes2net/scripts/validate_nes2net.py --device cpu
```

真实权重验证覆盖989项严格加载、真实/全伪造/部分伪造的单条评分、同实例重复调用、full模式最短最长输入、4s补齐及截取模式。报告在 `results/validation_cuda.json` 和 `results/validation_cpu.json`。单元测试验证输入约束、完整音频保留、原版补齐、原始分数方向、阈值边界、未校准时不判定、无定位输出等。

这验证实现可运行，不代表已经复现论文EER或证明泛化效果。没有自动加入五模型定位路由，也未更改网页默认模型。
