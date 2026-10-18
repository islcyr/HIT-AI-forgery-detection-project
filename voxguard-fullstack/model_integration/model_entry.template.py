"""复制至模型 source/model_entry.py，再按真实模型源码填写。

不得复制示例阈值作为已验证的模型阈值。不要返回模型类之外的报告对象。
"""


def create_model(**kwargs):
    # 替换为实际模型的导入和构造：
    # from actual_model.architecture import ActualNetwork
    # return ActualNetwork(**kwargs)
    # 如模型 forward 需要长度或其他固定参数，可返回一个 torch.nn.Module
    # 包装器；包装器输出必须给出真正的整段分数和真正的逐时间格分数。
    # 包装器改变 state_dict 的键名时，需提供与之严格匹配的权重文件。
    raise NotImplementedError("必须根据模型源码实现 create_model；不自动推测网络结构。")
