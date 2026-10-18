# 第三方来源与许可说明

本资料包收录 30 条从服务器已有数据副本复制的实际音频，来源为 PartialSpoof 与 LlamaPartialSpoof。源文件及原始标注未修改；新增接口文档和格式整理不改变第三方资料的许可条件。

## PartialSpoof

- 收录数量：20 条，真实语音和局部伪造各 10 条，均来自 v1.2 的 dev 集合。
- 作者：Lin Zhang、Xin Wang、Erica Cooper、Junichi Yamagishi、Jose Patino、Nicholas Evans。
- 原始版权声明：National Institute of Informatics，Copyright © 2021。
- 官方记录：[PartialSpoof v1.2](https://doi.org/10.5281/zenodo.5766198)。
- 数据及标签许可：官方 README 声明 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，并保留下述源数据库许可说明。
- 原始说明：[随包 README](samples/source/PartialSpoof_README_v1.2.txt)，保留作者、格式、引用、许可和免责声明。

音频取自已解压的 database/dev/con_wav，逐字节复制。原始 .vad 从 database_vad.tar.gz 按样例标识提取；文件内容未修改。官方协议按标识选取对应原始行。JSON 和 CSV 保留官方区间，并将未覆盖区域明确标为 unannotated；未推测该区域的真实性。

此前公开网页版中保留的 4 条样例已与本地数据集文件核对，字节一致。源数据库可能已有其发布流程中的归一化；本资料包整理过程未再次处理音频。

## ASVspoof 2019 源数据库

PartialSpoof 源自 ASVspoof 2019，其 README 保留源数据库的 [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1.0/index.html) 说明。

原始记录：[ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336)，[DOI](https://doi.org/10.7488/ds/2555)。原作者及贡献者的完整引用保留在 PartialSpoof README 中。本包的 LA_D_*.wav 是 PartialSpoof 发布版本中的真实 WAV，不是原始 ASVspoof FLAC。

## LlamaPartialSpoof

- 收录数量：10 条整段伪造，来自 v1.0.b 的 R01TTS.0.a 发布部分。
- 作者：Hieu-Thi Luong、Haoyang Li、Lin Zhang、Kong Aik Lee、Eng Siong Chng。
- 官方记录：[LlamaPartialSpoof v1.0.b](https://doi.org/10.5281/zenodo.14214149)。
- 数据许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
- 原始说明：[README](samples/source/LlamaPartialSpoof_README.txt) 与[完整许可](samples/source/LlamaPartialSpoof_LICENSE.txt)。

音频逐字节复制，原始标签行来自 label_R01TTS.0.a.txt。该文件的 MD5 已与官方发布页核对。选取的 full 样例均有官方完整时间区间；标签只区分真实与伪造，没有独立非语音标注。原始文本保留不变；标签四位小数带来的末端误差在派生 JSON 中对齐 WAV 实测时长，并逐条记录调整。

生成器信息保留样例标识中的原始名称。reference_speaker_key 仅为发布标识中的参考语料键，不声称识别实际说话人身份。R01TTS.0.a 是发布部分，不改写为开发集或训练集。

## 修改说明与再分发

本次仅选择样例、整理目录、复制音频、摘录原始标签和产生 JSON/CSV。原始数据目录不变，音频复制前后 SHA-256 一致。公开包记录数据集相对路径及校验值，不依赖服务器目录结构。

再次分享时应保留作者、数据集名称、来源、许可链接、原始说明及上述修改记录，不得暗示数据集作者或机构对本项目提供背书。正式研究应遵守原数据集的引用要求；主要参考如下。

1. Zhang, L. et al. An Initial Investigation for Detecting Partially Spoofed Audio. Interspeech 2021. [DOI](https://doi.org/10.21437/Interspeech.2021-738)。
2. Wang, X. et al. ASVspoof 2019: A Large-scale Public Database of Synthesized, Converted and Replayed Speech. Computer Speech & Language, 2020. [DOI](https://doi.org/10.1016/j.csl.2020.101114)。
3. Zhang, L. et al. The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance. IEEE/ACM TASLP, 2023. [DOI](https://doi.org/10.1109/TASLP.2022.3233236)。
4. Luong, H.-T. et al. LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation. [作者论文](https://arxiv.org/abs/2409.14743)。

HAD、ADD 等资源仅作为其他候选列于数据说明，未包含在当前压缩包中。
