# 检测与定位演示数据清单

核验日期：**2026-10-13**。当前资料包已从服务器已有的 PartialSpoof 与 LlamaPartialSpoof 数据副本选取 30 条音频，真实、整段伪造、局部伪造各 10 条，全部配有官方时间标注依据。具体目录见[音频样例说明](../samples/README.md)，字段解释见[标签指南](label-guide.md)。本次无需下载新增音频，未执行模型推理。

## 1. 数据来源

| 数据源与官方入口 | 语言、适用任务与标注 | 获取情况、许可与限制 |
|---|---|---|
| **[PartialSpoof v1.2 下载](https://zenodo.org/records/5766198)** · [项目](https://github.com/nii-yamagishilab/PartialSpoof) | 源自 ASVspoof 2019 LA 的英语语音；支持整段检测和时间定位。`.vad` 每行是 `start_time end_time label`，时间单位秒；`0=spoof`、`1=bonafide`、`2=non-speech`，另有多分辨率 `.npy` 标签。 | 已核实公开下载页：dev 音频约 **2.0 GB**，时间标签 `database_vad.tar.gz` 约 **10.2 MB**，协议约 **5.4 MB**。[数据 README](https://zenodo.org/records/5766198/files/README_v1.2?download=1) 写明 **CC BY 4.0**，并说明源库 ASVspoof 2019 为 ODC-By。当前正式 Zenodo 记录为 v1.2；页面另列 v1.3 临时时间标签并注明正式版本准备中，首批固定使用 v1.2。GitHub 代码许可不能替代数据许可。 |
| **[HAD 官方下载](https://zenodo.org/records/10377492)** · [作者论文](https://www.isca-archive.org/interspeech_2021/yi21_interspeech.pdf) | 中文普通话；提供局部伪造及篡改位置，适合中文检测与定位。 | 已核实公开 `HAD.zip`，约 **8.1 GB**。[目录预览](https://zenodo.org/records/10377492/preview/HAD.zip?include_deleted=0) 中含 train/dev/test、`HAD_*_label.txt`、README 和内部音频包。**标签列的精确格式及 HAD 数据许可尚未核实**，取样前先读包内 README；不能把源库 AISHELL-3 的许可直接套到 HAD。 |
| **[LlamaPartialSpoof 官方下载](https://zenodo.org/records/14214149)** · [项目](https://github.com/hieuthi/LlamaPartialSpoof) | 官方数据页标明英语；含真实、完全伪造、局部伪造，支持检测与定位。标签每行是 `id duration utterance_label segment...`；每个 segment 为 `start-end-label`，类别为 `bonafide/spoof`。 | 公开下载；两个音频包合计约 **28.2 GB**，标签可单独下载；官方写明 **CC BY 4.0**。原始版本按评测用途设计，后续研究另提供持续学习划分；研究划分和许可证是两回事。首批无需下载全量。 |
| **[ASVspoof 2019 下载](https://doi.org/10.7488/ds/2555)** · [官方主页](https://www.asvspoof.org/index2019.html) | 选 **LA**：英语 TTS/VC 整段检测，有语句级真伪、攻击类别等协议信息；**没有局部伪造时间真值**。[作者数据论文](https://www.pure.ed.ac.uk/ws/portalfiles/portal/157030755/ASVspoof_2019_WANG_DOA08052020_AFV.pdf) 说明英语 VCTK 来源。 | 已核实官方公开仓库入口，本次未下载核查归档内容。PartialSpoof 官方数据 README 将其源库许可注明为 **ODC-By**；获取时保留并复核原包许可。另一个 **real PA** 扩展集需要签协议，不属于此处推荐的 LA。 |
| **[ASVspoof 2021 DF 下载](https://zenodo.org/records/4835108)** · [官方标签与说明](https://www.asvspoof.org/index2021.html) | **整段检测**，包含 TTS/VC 和压缩条件；真伪 keys 与元数据单独提供。**没有局部定位真值**。本次查看的下载页未列完整语言清单，不据此保证中文覆盖。 | 已核实公开 DF **evaluation set**，约 **34.5 GB**、分 4 包；官方主页写明 **Open Data Commons Attribution Licence**。2021 没有新增训练/开发集，是挑战划分事实，不等于许可证禁止训练。首批可先不用此大包。 |
| **[ADD 2023 Track 2 官方下载入口](http://addchallenge.cn/downloadADD2023)** · [组织者论文](https://arxiv.org/html/2408.04967v1) | 中文语音来源；Track 2 为篡改区间定位。训练/开发的伪造部分来自 HAD；测试还含同一说话人真实录音替换、语音编辑，不能把所有篡改都解释为 AI 合成。 | 论文确认该下载入口，**本次官网访问超时，当前下载可用性、申请要求、标签文件格式和数据许可待核实**。保留作后续候选，不作为首批对接的前置条件。 |

## 2. 当前 30 条交付样例

| 类型 | 数量 | 来源与标签 |
| --- | ---: | --- |
| 真实语音 | 10 | PartialSpoof v1.2 dev，官方整段协议和原始 .vad；无伪造区间，保留真实、非语音及未标注区域 |
| 整段伪造 | 10 | LlamaPartialSpoof v1.0.b / R01TTS.0.a，官方 full 样例及全时长 spoof 区间，覆盖 6 个生成器标识 |
| 局部伪造 | 10 | PartialSpoof v1.2 dev，官方整段协议和原始 .vad；同时包含真实与伪造区间 |

全部为英语、16 kHz 单声道 16 位 PCM WAV，复制时不作音频变换。PartialSpoof 样例优先覆盖不同开发集说话人；Llama 的 reference_speaker_key 仅记录标识中的参考语料键，不代表合成说话人身份。两种来源的集合信息分别保留，不将 Llama 发布部分误写为开发集。

原 6 条包中的 4 条已标注样例保留，2 条缺少时间真值的整段伪造网页样例被有官方完整时间区间的样例替换。旧版 ZIP 保留，当前包总数为 30 条。

这些样例用于功能联调，不构成正式性能评测集。正式实验仍须按各基准划分执行，避免训练、调阈值和测试数据混用。后续如增加中文样例，优先核对 HAD 的具体标注和分发条件。损坏文件、纯静音或超长文件可作为独立异常输入测试，不计入本次 30 条有标签音频。

## 3. 后续扩充取样步骤

1. **先取说明与标签。** 从 PartialSpoof v1.2 页保存 README、许可说明、协议和 `database_vad.tar.gz`，确认原始 ID、整体标签与时间标签的对应关系；不应优先下载 train/eval 全量。
2. **再取一个较小的开发集包。** 下载 `database_dev.tar.gz`（仍约 2.0 GB）并校验官网给出的 MD5；只从 dev 中按官方标签挑选需要的样例。若已有合法取得的源包，直接从现有包取样，避免重复下载。
3. **按原始 ID 配对。** 音频、协议行、同名 `.vad` 必须属于同一条记录和兼容的数据版本。仅将 `.vad` 中 `0` 对应的区间映射为伪造区间，保留 `2` 的非语音语义；检查时间单位、起止顺序及是否超出音频时长。缺少时间真值时标记未提供，不得根据听辨、图示或整体标签推测。
4. **保留可追溯信息。** 记录原始 ID、来源页、数据版本、原始划分、语言依据、许可及核实状态；对实际交付的每个音频文件计算 SHA-256。保留原文件；若重采样或转码，记录处理步骤以及处理后文件的哈希，并确认时间轴仍能正确对应。
5. **输出统一清单。** 一行一条 JSONL，遵循 [样例清单 Schema](../contracts/sample-manifest.schema.json)，字段写法参考 [样例 JSONL](../examples/sample-manifest.jsonl)。样例清单中的真实标注与模型预测应分别保存；预测不能覆盖真值。
6. **再补其他类别和中文。** 补足 10 条完全伪造，再根据 HAD 包内说明加入中文记录。许可仍待确认的候选先保留来源记录，不加入公开分发的样例包。

只需先试听时，可用 [PartialSpoof 官方样例页](https://nii-yamagishilab.github.io/zlin-demo/IS2021/index.html)、[ASVspoof 2019 LA 官方样例页](https://www.asvspoof.org/audio_examples) 或 [LlamaPartialSpoof 官方样例页](https://hieuthi.github.io/sample-LlamaPartialSpoof/)。试听页不能代替样例与标签的正式配对；LlamaPartialSpoof 页面注明演示音频做过响度归一化，正式评测使用数据包。
