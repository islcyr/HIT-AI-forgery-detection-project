# SAL 单条音频推理

复用作者的 Segment-Aware Learning 网络和任务权重，无需训练。独立环境为项目根目录的 `.envs/sal`。前后端默认仍使用 CFPRF。

## 两个版本的实际状态

| 版本 | 编码器 | 状态 |
|---|---|---|
| **WavLM** | WavLM Large | **可用**：571 项状态严格加载，CPU / CUDA 单条音频推理通过 |
| **W2V2** | Wav2Vec2 XLS-R 300M | **官方任务权重损坏**：网络和编码器能初始化，但 checkpoint 截断，不能恢复完整训练参数 |

这是两种编码器版本，不是 CFPRF 那样的训练数据集选项。发布 checkpoint 元数据没有完整保存训练数据配置，接口不据文件名宣称具体训练集。

W2V2 文件的 SHA256 为 `bfa5e23c0500ba9bc720c3516a097233d6b1c4655994774d854b978b7add5c02`，大小 3357802496 字节，**与作者 Hugging Face 发布对象一致**。文件缺少 ZIP 尾部目录，PyTorch 报 `failed finding central directory`；部分模型张量的数据也不在现有文件中，不是仅丢失优化器信息。作者模型仓库没有其他历史版本，GitHub Releases 也无替代文件。

因此目前不能保证两个版本都能正常运行。代码拒绝损坏文件，不会以基础模型或随机参数填补缺失训练参数。取得完整 checkpoint 后可通过 `--checkpoint` 指定，仍进行 `strict=True` 校验。详见 [权重检查记录](results/checkpoint_audit.json) 和 [两版本验证报告](results/validation_cuda.json)。

## 单条调用

在**项目根目录**执行：

```bash
bash methods/sal/scripts/sal.sh voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --variant WavLM --device cuda:0 --include-frames \
  --output methods/sal/results/CON_D_0000029_sal_wavlm.json
```

CPU 使用 `--device cpu`。不传 `--output` 时标准输出为 JSON，依赖日志写入标准错误。

W2V2 接口已实现，但现有官方文件无法执行成功。取得完整权重后使用：

```bash
bash methods/sal/scripts/sal.sh your_audio.wav --variant W2V2 \
  --checkpoint /path/to/complete/SAL_W2V2.ckpt --device cuda:0 \
  --output methods/sal/results/result_w2v2.json
```

| 可选参数 | 默认值 | 用途 |
|---|---|---|
| `--variant` | `WavLM` | `WavLM` 或 `W2V2`；当前完整可用的发布权重是 WavLM |
| `--device` | `auto` | 自动选择 CUDA / CPU，或指定 `cuda:0`、`cpu` |
| `--threads` | `4` | PyTorch CPU 线程数 |
| `--threshold` | `0.5` | 帧级伪造分数阈值，严格大于时判为伪造 |
| `--include-frames` | 不启用 | 返回帧级伪造分数 |
| `--output` | 标准输出 | JSON 路径；自动创建父目录，同名文件覆盖 |
| `--checkpoint` | 所选版本的本地任务权重 | 指定完整可信的 SAL checkpoint |
| `--encoder-checkpoint` | 对应共享编码器权重 | 指定本地 SSL 编码器文件 |
| `-h` / `--help` | — | 查看帮助 |

### Python 接口

使用 `.envs/sal/bin/python`，在项目根目录运行：

```python
import sys
sys.path.insert(0, 'methods/sal')
from sal_runtime import SALDetector

model = SALDetector(variant='WavLM', device='cuda:0')  # 加载一次，重复调用
result = model.predict_file('your_audio.wav', include_frames=True)
# 或使用后端已解码的音频：
# result = model.predict(waveform, sample_rate=16000, include_frames=True)
```

输入一条 16 kHz 单声道浮点 PCM 音频，通常为一维 float32 数组。部署允许 400–340000 个采样点，即 0.025–21.25 秒；这是部署资源限制，不是 SAL 的论文时长上限。后端负责解码、重采样、声道转换及外部幅度预处理。本接口不重采样、不自动截断，保留模型内部原版 layer normalization。

## 输出与推理

沿用 CFPRF / BAM 的主要字段：`model`、`checkpoint_variant`、`sample_rate`、`duration`、`is_fake`、`label`、`score`、`score_type`、`decision_rule`、`frame_resolution`、`segments`，以及 SAL / BAM 使用的 `threshold`。每个片段含秒单位的 `start`、`end` 和 `score`。

- 原模型输出 8 类位置头和 2 类真假头，官方测试使用真假头，本接口同样使用真假头；第 0 类为伪造、第 1 类为真实。
- 输入末尾补到 2560 点（0.16 秒）的整数倍，保留网络内部的 256 点补零。单条推理不需要标签或数据集预处理。
- 对真假头取 softmax 的第 0 类分数，连续超过阈值的帧合并，末帧结束时间截到真实时长。
- 整段 `score` 是最大帧级分数，片段 `score` 是片段内最大帧级分数。这是部署后处理，不是校准后的概率，也不能直接与 CFPRF 的 PRN 分数比较。
- 定位分辨率为 0.16 秒；`--include-frames` 额外返回 `frame_fake_scores` 和 `frame_score_type`。

真实结果见 `results/CON_D_0000029_sal_wavlm.json`：预测伪造区间 0.32–2.24 秒。

## 环境与来源

本机环境已准备好。重新安装：

```bash
SAL_BASE_PYTHON="$(conda run -n py310 python -c 'import sys; print(sys.executable)')" \
  bash methods/sal/scripts/setup_sal.sh
```

Python 3.10、PyTorch / torchaudio 2.5.1+cu121、NumPy 1.23.5、s3prl 0.4.18。虚拟环境继承本机 py310 基础 PyTorch，独立安装推理依赖，见 `requirements-sal.txt`；不安装 Lightning、Hydra 和训练依赖。

源码：[SentryMao/SAL](https://github.com/SentryMao/SAL)，本地 `temp/SAL` 固定提交 `b4e80d1d2fd98f3d8b9a85a47837ab3a73e12485`。任务权重：[MaoYC/SAL](https://huggingface.co/MaoYC/SAL)，本地 `temp/weights/SAL/`。共享编码器在 `temp/weights/shared/`。

`sal_runtime/vendor/` 复制作者网络和 Attention / Conformer / Transformer 模块。适配仅调整导入、移除未使用的 Transformers 导入、将 s3prl 全量 hub 改为对应子模块；保留 25 层加权、Conformer、平均池化和输出头。共享 XLS-R 已使用普通字典配置，可按 s3prl 原转换的三字段格式生成 `xlsr2_300m.sal-converted.pt`，无需 Fairseq 训练环境。基础编码器初始化成功不代表已加载 SAL 训练权重。

## 验证

```bash
.envs/sal/bin/python -m unittest discover -s methods/sal/tests -v
.envs/sal/bin/python methods/sal/scripts/validate_sal.py --device cuda:0
```

验证脚本逐一尝试两个版本，写入 `results/validation_cuda.json`；**只要一个版本失败，就以非零状态退出**。W2V2 的失败原因会明确记录，随后继续验证 WavLM。

WavLM 已完成 571 项状态严格加载、CPU / GPU 单条推理、同一实例重复处理真实/部分伪造/全伪造音频，以及最短、最长和非整帧输入检查。5 项单元测试覆盖输入校验、幅度保留、区间合并、阈值、末帧截断及损坏权重报错。尚未复现论文完整测试集指标。
