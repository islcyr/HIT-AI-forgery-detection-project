"""非通用 PyTorch 波形模型的适配边界；复制到单独模块后实现。"""
from voxguard.adapters import AdapterError, AudioInput, Prediction


class CustomAdapter:
    def __init__(self):
        # 在这里导入源码、校验权重和配置、加载模型并切换到 eval。
        # 加载失败必须阻止启动，禁止伪造 ready 或退回模拟分数。
        raise NotImplementedError("请实现模型加载与版本绑定。")

    def describe(self):
        # 返回 contracts/openapi.yaml 的 Model 对象，包括 is_mock。
        # version 绑定权重、预处理、阈值及后处理；真实模型 is_mock=false。
        # 定位粒度必须是模型有效粒度，不使用插值伪造精度。
        raise NotImplementedError("请填写真实模型能力和输入限制。")

    def predict(self, audio: AudioInput, mode: str, include_timeline: bool) -> Prediction:
        # audio.waveform: float32, [channels,samples]，时间起点为上传音频起点。
        # 真正执行模型；重采样、分块、去填充、窗口重叠融合及时间映射须在此完成。
        # 定位成功时 timeline 连续覆盖 [0,duration_sec]，不重叠、不留空隙；
        # segments 的 spoof_score 为对应时间格按覆盖时长加权的均值。
        # model.predict 的分数必须来自音频，不读取样本标签/文件名推断答案。
        # 无定位能力：describe.localization=false，仅允许 detect，segments=null。
        # 失败：raise AdapterError("INFERENCE_FAILED", "模型推理失败")。
        raise NotImplementedError("请实现真实推理及统一结果转换。")


def create_adapter():
    return CustomAdapter()
