# 音频标签读取与展示说明

本资料包包含 30 条实际音频：真实语音、整段伪造和局部伪造各 10 条。标签用于说明样例的来源和官方标注，可用于上传、播放、时间轴与结果对照联调；不代表已执行模型推理。

## 文件入口

以下路径均相对于资料包根目录；解压后应保留目录结构。

| 文件或目录 | 内容与用途 |
| --- | --- |
| `samples/manifest.jsonl` | 30 条样例的总入口，每行是一个独立 JSON 对象；通过 `detailed_label_path` 找到详细标签 |
| `samples/audio/` | 可上传、播放的 WAV 音频；具体路径以清单为准 |
| `samples/labels/detailed/<sample_id>.json` | 单条音频的完整元数据、逐区间标签、原始来源与统计 |
| `samples/labels/summary.csv` | 每条音频一行的总体信息，便于筛选类别和查看时长 |
| `samples/labels/segments.csv` | 每个时间区间一行的明细，便于检查片段边界 |
| `samples/labels/original/` | 保留的 PartialSpoof 原始 `.vad` 文件及 Llama 每条样例的原始标签行 |
| `samples/labels/protocols/` | 官方协议中与本包样例对应的原始行 |
| `samples/source/provenance.json` | 源数据集相对路径、来源信息与校验值；不包含服务器绝对路径 |
| `samples/source/` | 数据集原始说明和许可材料 |

两个 CSV 使用带 BOM 的 UTF-8 编码，便于 Excel 识别。程序读取时可采用 `utf-8-sig`；时间字段应按数值处理，不要依赖电子表格显示的小数位数。

## 详细 JSON 的字段

| 字段 | 含义 |
| --- | --- |
| `schema_version`、`sample_id` | 标签结构版本及包内唯一样例标识 |
| `category` | 三分类：`bonafide` 为真实，`fullspoof` 为整段伪造，`partialspoof` 为局部伪造 |
| `utterance_label` | 整段二分类：`bonafide` 或 `spoof`；局部伪造也属于 `spoof` |
| `audio` | 包内路径、实际时长、采样率、声道数、采样帧数、位深及音频 SHA-256 |
| `source` | 数据集、版本、公开来源、原始标识、集合信息、攻击类型及协议原文等 |
| `annotation` | 时间单位、区间约定、标注来源、原始标签校验值、边界调整记录及覆盖时长 |
| `segments` | 按时间排列的逐区间标签，同时保留原始标签和原始起止时间 |
| `summary` | 各标签的区间数与总时长，以及伪造时长占比 |

`audio.duration_sec` 按实际 WAV 采样帧数与采样率计算。`audio.num_samples` 表示每声道的采样帧数。播放时间轴采用此实际时长，不使用文件名或标注文本中的近似时长。

`source` 中的未知字段使用 `null` 等明确值，不应补写推测的信息。Llama 的 `original_split=not_defined`：文件名中的 `dev01` 不代表官方开发集划分；`release_part=R01TTS.0.a` 记录数据发布部分。其 `speaker_id=null`，`reference_speaker_key` 仅为从原始标识取得的参考键，不表示合成音频的实际说话人身份。

## 时间区间含义

所有区间以原始音频起点为 0，单位为秒，采用左闭右开 `[start_sec, end_sec)`。区间时长为 `end_sec - start_sec`。展示时可四舍五入，定位和计算时应保留原始精度。

| `segments[*].label` | 含义 | 建议展示 |
| --- | --- | --- |
| `bonafide` | 官方标注的真实区间 | 默认颜色或绿色 |
| `spoof` | 官方标注的伪造区间 | 红色 |
| `non_speech` | 官方单独标出的非语音区间 | 灰色 |
| `unannotated` | 官方时间标注未覆盖的区间 | 留白或单独标记“未标注” |

每段的 `source_label`、`source_start_sec`、`source_end_sec` 保留原始标签与边界；包内补充的未标注区间没有对应的原始标签。不要将 `unannotated` 当作真实、伪造或非语音。

PartialSpoof 原始 `.vad` 中，`0=spoof`、`1=bonafide`、`2=non_speech`。个别音频的末尾存在不足 1 ms 的未覆盖区间，详细 JSON 显式记录为 `unannotated`，不延长官方标签；原始 `.vad` 文件保持不变。

Llama 的原始时间标签只区分真实与伪造，没有独立的非语音标注，因此 `non_speech_annotation_available=false`，清单中的 `non_speech_segments=null`。这表示“没有这类标注”，不表示音频没有静音。原始标签端点保留四位小数，与 WAV 实测时长相差不超过 0.00005 秒时，包内标签将末端对齐实际时长，并在 `annotation.normalizations` 记录；原始标签文件不修改。

`annotation.status=official_temporal_annotations` 表示时间标签来自官方材料，不表示全部时长均已标注。实际覆盖情况应读取 `annotated_duration_sec` 和 `unannotated_duration_sec`。

## 前端使用步骤

1. 按行解析 `manifest.jsonl`，使用 `sample_id` 关联音频、类别及详细标签；不要把整个文件当作一个 JSON 数组解析。
2. 通过清单中的音频路径加载 WAV，通过 `detailed_label_path` 加载详细 JSON；相对路径以资料包根目录为基准解析。
3. 以 `audio.duration_sec` 设置时间轴总长度，筛选 `segments` 中 `label=spoof` 的区间标红，界面标注“官方标注”。
4. 对宽度为 `W` 的时间轴，片段左边界为 `start_sec / duration_sec × W`，宽度为 `(end_sec - start_sec) / duration_sec × W`。
5. 点击片段时，将播放器 `currentTime` 设置为 `start_sec`；若实现片段播放，在到达 `end_sec` 时暂停。保留音频起始静音，不对播放文件另行裁剪。
6. 接入后端后，上传音频并根据 API 返回显示模型分数与预测片段。官方标签与模型预测应分开显示，分别注明来源。

静态页面可在同一站点提供解压后的目录，便于浏览器读取 JSON 和音频。接口流程与字段约定见 [API 说明](api-contract.md)，模型输入输出要求见 [模型对接说明](model-handoff.md)。

## 统计与结果的边界

`summary.spoof_ratio_total_duration` 为伪造标注时长除以音频总时长；`summary.spoof_ratio_annotated_speech` 为伪造标注时长除以真实与伪造标注时长之和。后者在具有独立非语音标注时计算，不包含 `non_speech` 和 `unannotated`；缺少非语音标注或分母为零时为 `null`。

Llama 不提供独立非语音标注，其真实或伪造区间可能包含静音，因此本包将 `spoof_ratio_annotated_speech` 设为 `null`，不报告未经验证的语音时长占比。各标签的区间数也不等于合成次数或攻击次数。

伪造时长占比是标注统计，不是模型分数、可信度或伪造概率。不要因为整段标签为 `spoof`，就自行补出覆盖全时长的定位结果；时间轴应始终使用逐区间标签。

[接口返回示例](../examples/) 中的分数和任务结果为模拟数据，与这些音频的真实推理结果没有对应关系。可用它们搭建结果页面，但不能将其与官方标签拼接后宣称为模型检测结果。

该资料包用于功能联调与人工检查，样例数量及选取方式不支持据此报告模型总体准确率或正式性能结论。来源与使用条件见 [第三方来源及许可](../THIRD_PARTY_NOTICES.md)。
