# 30 条音频及标签使用说明

本目录包含 30 条实际音频，真实、整段伪造、局部伪造各 10 条。所有音频均从服务器已有的数据集副本复制，没有重新下载音频，也没有裁剪、变速、重采样或幅度归一化。全部为英语、16 kHz、单声道、16 位 PCM WAV，总时长约 131.53 秒。

真实和局部伪造各 10 条来自 PartialSpoof v1.2 开发集；整段伪造 10 条来自 LlamaPartialSpoof v1.0.b 的 R01TTS.0.a 发布部分，覆盖 6 个生成器标识。LlamaPartialSpoof 的发布部分和文件名 dev01 不作为训练或开发集划分解释。

## 最快使用方式

1. 打开 [labels/summary.csv](labels/summary.csv)，筛选所需类别并查看文件名、时长和伪造区间数量。
2. 在 [audio](audio/) 中播放同名 WAV 文件。
3. 在 [labels/detailed](labels/detailed/) 中打开同名 JSON，查看每个区间的起止秒数、真假、非语音或未标注状态。
4. 批量检查时间区间时打开 [labels/segments.csv](labels/segments.csv)；需要核对原始依据时查看 [labels/original](labels/original/) 和 [labels/protocols](labels/protocols/)。
5. 前端或后端程序按行读取 [manifest.jsonl](manifest.jsonl)，通过 audio_path 和 detailed_label_path 关联音频与标签。文件中的相对路径以资料包根目录为基准，不以当前 samples 目录为基准。

两个 CSV 使用 UTF-8 BOM 编码，便于电子表格软件读取。界面可以缩短小数显示，计算时应保留数据精度。

## 30 条样例目录

点击样例标识可打开其详细标签；音频位于 audio/ 下的同名 WAV。

| 样例标识 | 类别 | 时长 秒 | 伪造区间数 | 数据来源 |
| --- | --- | ---: | ---: | --- |
| [LA_D_1024892](labels/detailed/LA_D_1024892.json) | 真实 | 2.131562 | 0 | PartialSpoof |
| [LA_D_5868407](labels/detailed/LA_D_5868407.json) | 真实 | 2.858375 | 0 | PartialSpoof |
| [LA_D_1026868](labels/detailed/LA_D_1026868.json) | 真实 | 5.374937 | 0 | PartialSpoof |
| [LA_D_1028589](labels/detailed/LA_D_1028589.json) | 真实 | 4.088750 | 0 | PartialSpoof |
| [LA_D_1038259](labels/detailed/LA_D_1038259.json) | 真实 | 4.686063 | 0 | PartialSpoof |
| [LA_D_1047731](labels/detailed/LA_D_1047731.json) | 真实 | 5.414062 | 0 | PartialSpoof |
| [LA_D_1049549](labels/detailed/LA_D_1049549.json) | 真实 | 3.535062 | 0 | PartialSpoof |
| [LA_D_1060968](labels/detailed/LA_D_1060968.json) | 真实 | 4.094812 | 0 | PartialSpoof |
| [LA_D_1072893](labels/detailed/LA_D_1072893.json) | 真实 | 4.602375 | 0 | PartialSpoof |
| [LA_D_1084356](labels/detailed/LA_D_1084356.json) | 真实 | 2.683312 | 0 | PartialSpoof |
| [dev01-cosyvoice-full_6313_66129_000092_000001](labels/detailed/dev01-cosyvoice-full_6313_66129_000092_000001.json) | 整段伪造 | 7.883188 | 1 | LlamaPartialSpoof |
| [dev01-elevenlab-full_1988_24833_000025_000000](labels/detailed/dev01-elevenlab-full_1988_24833_000025_000000.json) | 整段伪造 | 5.433500 | 1 | LlamaPartialSpoof |
| [dev01-gptsovits-full_6345_93306_000056_000002](labels/detailed/dev01-gptsovits-full_6345_93306_000056_000002.json) | 整段伪造 | 14.860000 | 1 | LlamaPartialSpoof |
| [dev01-ljjets-full_2277_149896_000003_000001](labels/detailed/dev01-ljjets-full_2277_149896_000003_000001.json) | 整段伪造 | 9.705937 | 1 | LlamaPartialSpoof |
| [dev01-xttsv2-full_1462_170138_000003_000005](labels/detailed/dev01-xttsv2-full_1462_170138_000003_000005.json) | 整段伪造 | 2.506687 | 1 | LlamaPartialSpoof |
| [dev01-yourtts-full_6241_61943_000003_000000](labels/detailed/dev01-yourtts-full_6241_61943_000003_000000.json) | 整段伪造 | 2.113000 | 1 | LlamaPartialSpoof |
| [dev01-cosyvoice-full_1272_141231_000034_000002](labels/detailed/dev01-cosyvoice-full_1272_141231_000034_000002.json) | 整段伪造 | 2.809625 | 1 | LlamaPartialSpoof |
| [dev01-elevenlab-full_8297_275154_000024_000008](labels/detailed/dev01-elevenlab-full_8297_275154_000024_000008.json) | 整段伪造 | 4.049000 | 1 | LlamaPartialSpoof |
| [dev01-gptsovits-full_2902_9008_000032_000001](labels/detailed/dev01-gptsovits-full_2902_9008_000032_000001.json) | 整段伪造 | 9.020000 | 1 | LlamaPartialSpoof |
| [dev01-ljjets-full_2803_154328_000069_000000](labels/detailed/dev01-ljjets-full_2803_154328_000069_000000.json) | 整段伪造 | 5.015500 | 1 | LlamaPartialSpoof |
| [CON_D_0000000](labels/detailed/CON_D_0000000.json) | 局部伪造 | 1.976250 | 2 | PartialSpoof |
| [CON_D_0023779](labels/detailed/CON_D_0023779.json) | 局部伪造 | 1.899875 | 1 | PartialSpoof |
| [CON_D_0000001](labels/detailed/CON_D_0000001.json) | 局部伪造 | 5.004625 | 2 | PartialSpoof |
| [CON_D_0000002](labels/detailed/CON_D_0000002.json) | 局部伪造 | 3.644500 | 1 | PartialSpoof |
| [CON_D_0000004](labels/detailed/CON_D_0000004.json) | 局部伪造 | 1.980125 | 1 | PartialSpoof |
| [CON_D_0000008](labels/detailed/CON_D_0000008.json) | 局部伪造 | 4.318187 | 1 | PartialSpoof |
| [CON_D_0000015](labels/detailed/CON_D_0000015.json) | 局部伪造 | 2.868125 | 1 | PartialSpoof |
| [CON_D_0000019](labels/detailed/CON_D_0000019.json) | 局部伪造 | 1.820875 | 1 | PartialSpoof |
| [CON_D_0000029](labels/detailed/CON_D_0000029.json) | 局部伪造 | 3.104125 | 1 | PartialSpoof |
| [CON_D_0000031](labels/detailed/CON_D_0000031.json) | 局部伪造 | 2.047750 | 1 | PartialSpoof |

## 标签完整性与来源

全部 30 条均有官方时间标签或官方整段覆盖区间，未使用模型输出作为标签。PartialSpoof 提供真实、伪造、非语音三类时间区间；官方标注未覆盖的间隙和不足 1 ms 的末尾，在详细标签中明确标为 unannotated。Llama 的本次 10 条全伪样例由官方明确标注为全时长 spoof；其非语音标注缺失，因此 non_speech_segments=null，不表示没有静音。

Llama 原标签时长保留四位小数，与 WAV 实测时长差值不超过 0.00005 秒；仅在派生 JSON 中将全伪区间末端对齐实际时长，调整值记录在 annotation.normalizations。原始标签文件未修改。该末端对齐不改变音频。

- [标签读取与展示说明](../docs/label-guide.md)：各字段、类别、统计指标和前端标红方法。
- [详细标签 Schema](../contracts/detailed-label.schema.json)：逐条 JSON 的结构约束。
- [来源与文件校验](source/provenance.json)：数据集相对路径、来源文件校验及整理记录。
- [第三方来源与许可](../THIRD_PARTY_NOTICES.md)：作者、使用条件和引用要求。

与前一版 6 条资料包相比，保留了 4 条已有完整时间标签的样例，原来 2 条缺少时间标签的整段伪造网页样例由带官方全时长标注的 Llama 样例替换。旧版压缩包另行保留。

## 展示与模型接入

标签预览使用 detailed JSON 中 label=spoof 的时间区间，并注明“官方标注”。模型接入后使用 API 返回的预测区间显示“模型预测”；两者应独立保存和展示。[examples](../examples/) 是独立的模拟接口结果，不是这些音频的实测结果。

样例用于上传、播放和时间轴联调，不构成正式性能测试集。跨数据集类别分布、共享说话人及人工抽样均可能造成偏差，不能据此报告模型总体准确率。
