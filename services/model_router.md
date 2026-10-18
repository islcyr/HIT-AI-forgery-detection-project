# 五版本统一推理入口

`services/model_router.py` 提供单条音频的统一 Python / 命令行接口。固定运行 CFPRF-HAD、CFPRF-PS、CFPRF-LAVDF、BAM-WavLM、SAL-WavLM 五个版本，输出一份最终结果。SAL-W2V2 不参与。

## 调用

在项目根目录：

```bash
python services/model_router.py voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --device cuda:0 --output results—temp/CON_D_0000029_ensemble.json
```

路由本身只使用 Python 标准库，不需要安装第四套模型环境。它通过各自 `.envs/cfprf`、`.envs/bam`、`.envs/sal` 的 Python 启动推理进程。CPU 使用 `--device cpu`；`auto` 为默认设备选项。`--threads` 默认 4；`--timeout` 默认 300 秒，表示**每个模型**的超时时间。不指定 `--output` 时输出 JSON 到标准输出；指定时自动创建父目录，并在全部成功后原子替换同名结果文件。

Python 调用：

```python
from services.model_router import ModelRouter

router = ModelRouter(device='cuda:0', threads=4, timeout=300)
result = router.predict_file('your_audio.wav')
```

输入一条已经预处理好的音频文件：16 kHz、单声道，0.025–21.25 秒。路由不重采样、不归一化、不切窗、不训练。现有网页后端的最小时长仍为 0.5 秒。

## 融合规则

这是一个明确可复现的初始部署规则，尚未经过完整验证集的阈值或融合策略选择：

1. 读取五份模型输出中的伪造片段，保持各自的原始定位边界。
2. 用所有片段边界划分时间轴，分别统计每个区间被几个版本标为伪造。一个版本即使有重叠候选，也只计一票。
3. **至少 3/5 个版本在同一时间区间标为伪造**，该区间才进入最终结果；将连续通过的区间合并。
4. 存在最终片段则整段 `label=fake`，否则为 `real`。这不是对整段真假标签单独投票：多个模型判为伪造但定位完全不重合时，仍可能没有最终片段。

不对 PRN 分数与 softmax 分数求平均。最终 `score` 是时间轴上的**最高投票支持比例**，取值 0、0.2、0.4、0.6、0.8、1；片段 `score` 是片段内最高支持比例。它不是伪造概率，也不是单模型的检测分数。每个时间小段的详细支持情况见 `vote_timeline`。

CFPRF 三套权重各占一票，因此同一种方法可以占 3/5 票；这些投票并非独立证据。当前规则不保证优于单模型，正式选择仍应依据目标场景验证集。

## 输出

最终 JSON 包含：

- `model=ensemble-5`、`fusion=temporal_majority_vote`。
- `sample_rate`、`duration`、`is_fake`、`label`、`score`、`score_type`、`decision_rule`。
- `models_total=5`、`votes_required=3`。
- `segments`：最终伪造片段，字段为 `start`、`end`、`score`，时间单位秒。
- `vote_timeline`：每个时间区间的 `fake_votes`、`score`、`supporting_models`。
- `model_results`：以 `cfprf-had`、`cfprf-ps`、`cfprf-lavdf`、`bam-wavlm`、`sal-wavlm` 为键，保存五份原始推理结果。路由不请求单模型的帧分数，避免扩大输出文件。

五个模型必须全部成功，且音频时长、采样率、版本标识和区间格式一致。任一失败或超时，抛出 `RouterError` / 命令以状态 1 退出，不会自动减少投票分母或生成部分融合结果；已有结果文件保持不变。

## 运行方式与验证

当前五个进程依次运行，每个进程运行后释放模型和显存。每次路由调用都重新加载权重，整体耗时包含五次推理和加载；不是五模型常驻服务。网页已接入五模型常驻协调服务，调用本文件的同一融合函数，五个模型只在启动时加载。网页/常驻API说明见 ../voxguard-fullstack/docs/ensemble-resident.md；本命令行入口仍是离线逐次加载模式。

```bash
python -m unittest discover -s services/tests -v
```

测试覆盖时间重合投票、非重合片段、单模型重叠候选、连续区间合并、缺失/异常输出、五版本调用和失败/超时处理。真实五版本推理结果示例见项目根目录 `results—temp/CON_D_0000029_ensemble.json`。
