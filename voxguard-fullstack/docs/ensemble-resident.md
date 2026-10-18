# 五模型常驻接入

在项目根目录执行 `bash start.sh`，网页与 API 仍为 `http://127.0.0.1:8000`。页面、上传流程和用户选择项均未修改。前端自动获取唯一可用的综合检测描述，上传后后端自动调用 CFPRF-HAD、CFPRF-PS、CFPRF-LAVDF、BAM-WavLM、SAL-WavLM。

停止：`bash start.sh --stop`。设备可指定 `bash start.sh --device cuda:1` 或 `--device cpu`。五个版本默认都加载到指定设备，每个版本使用原来的独立环境；GPU需能同时容纳五套模型。首次启动校验权重并加载模型，后续任务复用进程与权重，不再逐任务加载。

## 调用链

网页 → 原有 FastAPI 上传/解码/重采样 → ensemble_resident 适配器 → 本机8765端口协调服务 → 五个常驻推理进程 → 现有 `fuse_results` → 原有结果协议 → 网页。

协调服务 `services/serve_ensemble.py` 是标准库HTTP服务，仅监听127.0.0.1；五个 `services/ensemble_worker.py` 使用各自环境，通过stdio JSON-lines通信。进程在启动时加载一次，随后对每条请求直接调用原适配器的 `predict()`。推理顺序执行，由同一个锁保护，避免五模型同时推理产生显存峰值和任务混线。协调服务关闭时清理五个子进程；任一工作进程退出或超时，不返回部分融合，不自动退化到少数模型投票。

既有命令行路由 `python services/model_router.py ...` 保留为离线逐次加载的调用方式。需要复用常驻模型时使用启动后的网页或：

```bash
.envs/voxguard/bin/python services/analyze.py your_audio.wav --output result.json
```

## 融合与输出

沿用原来的同一时间区间至少3/5票规则，没有加入TFCL或Nes2Net。真假判定、分数和定位均来自综合结果；分数是最大投票支持比例，不是校准概率。用户不需要选择五个模型中的任何一个。

后端协议不变，模型ID为 `ensemble-five`。定位以原始区间边界融合，描述中的0.16秒是参与模型中较粗的名义分辨率。返回连续的内部投票时间线及融合片段；片段分数保留该片段内最大支持比例，以原有proposal聚合类型传递，不伪装成时间平均值。原API根据include_timeline决定是否将内部时间线返回客户端。

输入限制沿用网页0.5–21.25秒、解码后16kHz单声道。后端原有音频格式处理与资源限制继续生效。

## 健康检查与版本

`http://127.0.0.1:8765/health` 返回协调服务PID、五个工作进程PID、各自 `model_load_count=1` 和已完成请求数；协调服务总加载数为5。多条音频测试时这些PID和加载次数应保持不变，仅请求计数增加。

`model_integration/ensemble-resident.json` 绑定权重、五种适配代码与融合代码的哈希，防止运行版本与描述不符。修改相关实现后重新生成：

```bash
python services/configure_ensemble.py
```

然后停止并重新启动服务。`.env` 和启动器默认工厂已切到 `model_integration.ensemble_resident:create_adapter`，没有启用模拟模型。旧CFPRF单模型接入文件保留用于单方法调用。
