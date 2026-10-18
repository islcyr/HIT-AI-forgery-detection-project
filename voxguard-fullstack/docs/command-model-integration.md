# CFPRF 与其他命令行模型接入

## 运行位置与配置

本次实现统一输出协议、可配置命令执行器和 CFPRF 输出转换器；不包含远端 HTTP 推理服务。最快联调方式是在模型服务器运行整个 VoxGuard 工程。界面可通过 SSH 隧道访问服务。若后端留在本地，需要另外实现 HTTP 传输适配器，它仍可复用本次输出转换函数；无需更改模型原生输出。

复制 `model_integration/cfprf-command.example.json` 为 `cfprf-command.json`，修改 `cwd` 为服务器模型项目绝对路径，并填写真实权重及代码版本。输入时长范围需根据实际模型核实。根目录 `.env` 设置：

```dotenv
ENABLE_MOCK=false
VOXGUARD_ADAPTER_FACTORY=model_integration.command_adapter:create_adapter
VOXGUARD_COMMAND_CONFIG=model_integration/cfprf-command.json
```

在安装后端依赖和 FFmpeg 的环境中运行 `python run.py`。模型脚本继续使用自己的 `.envs/cfprf/bin/python`，后端不需要导入模型/PyTorch。启动检查配置和必需文件，不进行 GPU 热身；目录 ready 表示可提交任务，GPU 或权重运行错误会使任务失败。每次任务创建独立临时输入和输出，使用参数数组执行命令，不拼接 Shell。日志写入 worker.log，超时或非零退出返回明确失败，模型原始结果文件读取后随临时目录删除。

输入按配置采样率重采样为单声道 IEEE float32 WAV，保持幅度和时间原点。需确认模型音频读取器支持 float WAV。命令必须为接收输入和输出路径的有限任务；示例 sh 使用 exec 启动 Python，执行器超时终止该进程。后续方法不要用脚本启动脱离管理的后台推理进程。

## CFPRF 语义

保留候选片段起止和验证分数，不补造片段之外的分数。定位结果为 `score_aggregation=proposal`，`timeline=null`；网页展示原生候选分数及严格 `> 0.5` 的整段阈值。分数不是经过校准的概率。

转换器核验 CFPRF、配置的权重变体、采样率、时长、分辨率和判定规则。当前约定假音 label=fake、真音 label=real，与 is_fake 一致；真音 label 的实际拼写尚需服务器输出确认。所有原生字段必须提供，包括空候选时的 segments=[] 和有限 score。若无候选时输出 null/缺字段，应明确其模型语义后调整转换器，不默认补零。

片段按时间排序；重叠、越界、非有限分数被拒绝，避免任意合并改变模型含义。如果真实 CFPRF 会输出重叠候选，应先确定 NMS 或片段融合策略并绑定版本。`frame_resolution` 当前映射为声明的定位分辨率，需要在服务器确认它是否代表最终候选边界粒度。

## 新方法接口

配置中的 models 可包含多个模型，每个条目独立指定 descriptor、command、output_converter 和 converter_options。工厂返回适配器列表，网页沿用已有模型选择功能；模型 id 必须唯一。尚未实现多模型同时推理及融合。

不同原始 JSON 格式通过独立转换函数处理，不让前端猜测字段名：

```python
from voxguard.adapters import Prediction

def convert_output(raw, audio, descriptor, mode):
    # raw：模型原生 JSON；audio：原始时长等信息。
    # descriptor：模型描述及 converter_options；描述字段优先。
    # 转换标签、分数方向、时间单位；不得伪造缺失分数。
    return Prediction(result=..., analysis_sample_rate_hz=...)
```

配置 `output_converter=your_module:convert_output`。返回格式必须符合后端 Result；整段判定可使用 gt/gte，并与 descriptor.decision_operator 一致。候选分数使用 proposal；具有真实连续时间格的输出使用 timeline_mean，后者继续验证片段加权均值。现有 Adapter.predict 接口保留，特殊多输入或远端服务可直接实现 Adapter，不受命令执行器限制。

后续融合应使用单独的适配器/策略，保留参与模型的身份、版本和原始语义；未经校准的不同模型分数不能直接平均。
