# BAM 单条音频推理

使用作者公开的 **WavLM + PartialSpoof** 权重，无需训练。实现位于本目录，独立环境为项目根目录的 `.envs/bam`。已完成 CPU、CUDA 真实音频推理和模型实例重复调用；当前前后端默认仍使用 CFPRF。

## 直接调用

在项目根目录执行：

```bash
bash methods/bam/scripts/bam.sh voxguard-fullstack/samples/audio/CON_D_0023779.wav \
  --device cuda:0 --include-frames --output methods/bam/results/result.json
```

没有 GPU 时改为 `--device cpu`。不传 `--output` 时，标准输出为 JSON，依赖日志进入标准错误。输出文件父目录会自动创建，同名文件会覆盖。

Python 接口（使用 `.envs/bam/bin/python`）：

```python
import sys
sys.path.insert(0, 'methods/bam')
from bam_runtime import BAMDetector

detector = BAMDetector(device='cuda:0')  # 启动时加载一次
result = detector.predict_file('your_audio.wav', include_frames=True)
# 后端已经解码成一维 float32 数组时：
# result = detector.predict(waveform, sample_rate=16000, include_frames=True)
```

输入是一条 **16 kHz、单声道、浮点 PCM** 音频，部署允许 0.025–21.25 秒。采样率、声道和幅度预处理由调用方负责；本方法不会重采样、转换声道或裁掉超长输入。21.25 秒是为了约束图注意力显存开销设置的部署上限，不是论文固有限制。WavLM 自身所需的内部 layer normalization 保留原版行为。

## 输出

主要字段与 CFPRF 对齐：

```json
{
  "model": "BAM",
  "checkpoint_variant": "WavLM-PartialSpoof",
  "sample_rate": 16000,
  "duration": 1.9,
  "is_fake": true,
  "label": "fake",
  "score": 0.93,
  "score_type": "max_frame_softmax_fake_class_0",
  "decision_rule": "at_least_one_frame_with_fake_score_gt_threshold",
  "threshold": 0.5,
  "frame_resolution": 0.16,
  "segments": [{"start": 0.48, "end": 1.6, "score": 0.93}]
}
```

以上为格式示意。真实输出见 `results/`。`score` 是整条音频最大帧级伪造分数，片段 `score` 是该片段内最大帧级分数，均在 0–1 内；与 CFPRF 的候选片段验证分数含义不同，不能直接比较大小或当作校准后的概率。

默认将伪造分数 **大于 0.5** 的连续帧合并为片段，存在任一片段则判为 fake；`--threshold` 可修改阈值。这是部署后处理规则，论文模型直接输出帧级分类 logits 和边界分数。定位粒度为 **0.16 秒**，末帧结束时间截到实际时长。分类第 0 类为伪造、第 1 类为真实。

`--include-frames` / `include_frames=True` 额外返回：

- `frame_fake_scores`：每 0.16 秒一个伪造分数，最后一帧可不足 0.16 秒。
- `frame_score_type`：`softmax_fake_class_0_uncalibrated`。
- `frame_boundary_scores`：模型边界分支的输出；用于解释帧边界，与伪造分数不同。

末尾按 2560 个采样点的倍数补零，随后保留原模型内部的 256 点补零，与作者评估流程一致。部署推理不需要训练标签、边界标签或 PartialSpoof 数据预处理脚本。

## 环境和权重

本机环境已准备好。重新准备时：

```bash
BAM_BASE_PYTHON="$(conda run -n py310 python -c 'import sys; print(sys.executable)')" \
  bash methods/bam/scripts/setup_bam.sh
```

Python 3.10，PyTorch / torchaudio 2.5.1+cu121，NumPy 1.23.5，s3prl 0.4.15。环境与 CFPRF 分开，沿用本机 py310 基础环境中的 PyTorch；不依赖 Lightning、Fairseq 或其训练环境。只导入 s3prl 的 WavLM 子模块，安装时不拉取全量 hub 的可选依赖。依赖固定于 `requirements-bam.txt`。

权重已在本地，不需要额外手动下载：

- `temp/weights/BAM/model.ckpt`：作者下载文件，实际为 ZIP，SHA256 为 `5c7379d23028606d3beeb8b1aa5c340395fcc403ce70a2ca559ea322ab2994d1`。
- `temp/weights/BAM/extracted/model.ckpt`：真正的 Lightning checkpoint；由 `scripts/prepare_weights.py` 校验下载文件并解压。
- `temp/weights/shared/wavlm_large.pt`：WavLM Large 基础权重，与其他方法共享存储。

加载时移除 `model.` 前缀并执行 `strict=True`，完整恢复 688 项状态。checkpoint 中训练配置类被映射为普通 namespace，不需要导入训练代码。`--checkpoint` 只接受解压后的可信作者 checkpoint；`--wavlm-checkpoint` 可指定基础权重路径。

## 来源与验证

官方仓库：[media-sec-lab/BAM](https://github.com/media-sec-lab/BAM)，本地源码位于 `temp/BAM`，固定提交 `55f3fb9e3b4dd6281597b86d7712fb23454179f6`。

`bam_runtime/vendor/` 复制作者 `models/bam.py` 及其三个依赖模块，保留原计算逻辑。适配仅包括相对包导入，以及将 `s3prl.hub.wavlm_local(...)` 改为它实际调用的 `s3prl.upstream.wavlm.expert.UpstreamExpert(...)`，避免导入无关模型依赖。

```bash
.envs/bam/bin/python -m unittest discover -s methods/bam/tests -v
.envs/bam/bin/python methods/bam/scripts/validate_bam.py --device cuda:0
```

验证覆盖输入约束、片段合并、阈值、末帧截断，以及同一实例处理真实、全伪造、部分伪造样例和最短/最长部署输入。结果保存至 `results/validation_cuda.json`。这些是部署验证，尚未复现论文完整测试集指标。
