# 音频伪造检测与定位


前后端已接入五模型常驻综合检测（CFPRF三套权重、BAM、SAL）。**在本目录执行 `bash start.sh`**，打开 http://127.0.0.1:8000 ，API 文档为 `/docs`。默认使用 GPU 0，可加 `--device cpu`。再次启动前应先停止已有服务。

停止服务：`bash start.sh --stop`，或在启动终端按 Ctrl+C。

服务启动后，直接调用一条音频：

```bash
.envs/voxguard/bin/python services/analyze.py your_audio.wav --output result.json
```

客户端自动上传并等待结果。有效时长 0.5–21.25 秒，支持 WAV/FLAC/MP3。详细配置、独立环境和接口说明见 [五模型常驻接入](voxguard-fullstack/docs/ensemble-resident.md)。

每个方法的适配代码、依赖、脚本、样例、测试和结果放在 `methods/<方法名>/` 下。

五个可运行版本的统一入口为 [services/model_router.py](services/model_router.py)，一次运行 CFPRF 三套权重、BAM-WavLM、SAL-WavLM，按同一时间区间至少 3/5 票生成最终结果：

```bash
python services/model_router.py voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --device cuda:0 --output results—temp/CON_D_0000029_ensemble.json
```

最终结果同时保留五份原始输出。`score` 为投票支持率，不是模型概率。融合规则、Python 接口、失败处理及加载开销见 [路由说明](services/model_router.md)。当前网页自动调用五个常驻版本并返回综合结果，无需用户选择。

```text
audio-forgery/
├── README.md             # 项目总入口
├── methods/
│   ├── cfprf/            # CFPRF 实现及说明
│   ├── bam/              # BAM 单条音频推理及说明
│   ├── sal/              # SAL；WavLM 可用，W2V2 发布权重损坏
│   ├── tfcl/             # TFCL 整段真假检测
│   └── nes2net/          # Nes2Net-X 整段评分，阈值需校准
├── services/             # 双服务启动器、配置生成与调用客户端
├── voxguard-fullstack/   # 前端、后端及模型适配器
├── temp/                 # 下载的官方源码、权重及来源记录
├── third_party/          # 第三方框架，目前为固定版本 Fairseq
└── .envs/                # 各模型与后端 voxguard 独立环境
```

CFPRF 已验证三套官方权重的加载、单条语音推理和原版解码对照。详细接口与输入限制见 [CFPRF 说明](methods/cfprf/README.md)。

在项目根目录运行：

```bash
bash methods/cfprf/scripts/cfprf.sh methods/cfprf/examples/audio/speech.wav \
  --variant HAD --device cuda:0 --output methods/cfprf/results/result.json
```

`--variant` 是可选参数，支持 `HAD`、`PS`（PartialSpoof）、`LAVDF`，省略时默认 `HAD`。三套权重均已下载。比如用 PS 权重检测当前 PartialSpoof 样例：

```bash
bash methods/cfprf/scripts/cfprf.sh voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --variant PS --device cuda:0 --include-frames \
  --output results—temp/CON_D_0000029_cfprf_ps.json
```

此参数用于单条推理命令，不会更改网页常驻服务的权重配置。完整命令参数见 [CFPRF 说明](methods/cfprf/README.md#命令参数)。

新增方法时使用独立的 `methods/<方法名>/` 目录。公共源码、权重及环境仍统一管理，避免在根目录增加各方法的推理脚本和依赖文件。

BAM 已提供独立推理接口，使用本地 WavLM / PartialSpoof 权重。详见 [BAM 说明](methods/bam/README.md)。在项目根目录执行：

```bash
bash methods/bam/scripts/bam.sh audio/CON_D_0023779.wav \
  --device cuda:0 --include-frames --output methods/bam/results/result.json
```

输入为一条 16 kHz 单声道音频，输出真假判断、分数、伪造片段和可选帧级分数；网页已自动接入五模型常驻综合检测。

SAL 已提供单条推理接口，详见 [SAL 说明](methods/sal/README.md)。**WavLM 已验证 CPU / GPU 可用；W2V2 的作者发布 checkpoint 截断，目前不能完整加载。**

```bash
bash methods/sal/scripts/sal.sh voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --variant WavLM --device cuda:0 --include-frames \
  --output methods/sal/results/CON_D_0000029_sal_wavlm.json
```

`--variant` 支持 `WavLM`（默认）和 `W2V2`；后者需要通过 `--checkpoint` 提供完整权重。输入要求与 BAM 一致，两版本验证报告见 `methods/sal/results/validation_cuda.json`。

TFCL 已提供整段真假检测接口，详见 [TFCL 说明](methods/tfcl/README.md)：

```bash
bash methods/tfcl/scripts/tfcl.sh voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --device cuda:0 --output methods/tfcl/results/result.json
```

TFCL 输出整段真假与分数，不输出伪造时间区间。超过约4.04秒会分窗检测后取最大伪造分数，不会忽略音频尾部；这属于部署聚合规则。当前未加入五版本定位路由。

Nes2Net-X 已提供整段评分接口，详见 [Nes2Net 说明](methods/nes2net/README.md)：

```bash
bash methods/nes2net/scripts/nes2net.sh voxguard-fullstack/samples/audio/CON_D_0000029.wav \
  --device cuda:0 --output methods/nes2net/results/result.json
```

默认 `--test-mode full`，完整音频一次推理，输出作者原始真实logit（越高越偏真实）。尚未校准阈值，默认 `is_fake=null`、`label=unclassified`；取得校准阈值后可传 `--threshold`。本方法没有定位输出，也未接入当前五版本路由。
