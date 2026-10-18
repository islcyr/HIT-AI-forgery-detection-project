# 本机真实 CFPRF 接入

从外层项目根目录（包含 `start.sh` 的目录） 启动：

```bash
bash start.sh
```

启动器先检查文件哈希并加载常驻 CFPRF，再启动后端。网页地址 http://127.0.0.1:8000 ，API 文档 http://127.0.0.1:8000/docs 。模型服务仅绑定 127.0.0.1:8765。Ctrl+C 结束启动器时，后端和模型进程一起停止。

也可在另一个终端执行 `bash start.sh --stop`，停止本项目管理的两个服务。PID 记录在 `services/.voxguard.pid`，停止前会核对它确实属于本项目启动器。

可指定 `--device cpu`、`--device cuda:1`、`--port 8001`、`--model-port 8766`；修改模型端口时，启动器自动为后端设置实际地址。端口已被占用时明确报错。

## 一条音频直接调用

服务启动后，在另一个终端的外层项目根目录执行：

```bash
.envs/voxguard/bin/python services/analyze.py audio.wav --output result.json
```

该客户端负责上传、轮询和保存完整 API 返回，默认 `detect_and_localize`；`--mode detect` 仅返回整段判定。`result.json` 是前后端协议的完整任务结果，不是原生 CFPRF JSON。检测结果位于 `result.decision`，伪造区间位于 `result.localization.segments`。

也可以直接使用 HTTP：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyses \
  -F 'file=@audio.wav' -F 'model_id=cfprf-had' -F 'mode=detect_and_localize'
curl http://127.0.0.1:8000/api/v1/analyses/返回的analysis_id
```

## 环境与流程

- 后端：外层 `.envs/voxguard`，Python 3.13 / NumPy 2.1.3，准确依赖见 `backend/requirements-lock.txt`。
- 模型：外层 `.envs/cfprf`，Python 3.10 / NumPy 1.23.5 / PyTorch 2.5.1+cu121。
- 服务默认仅注册 `cfprf-had`，`is_mock=false`；真实模型启动失败时不会回退到模拟数据。
- 后端保留原有队列和任务进程。任务进程只负责解码、等权声道混合、FFmpeg 重采样和结果转换，通过本机 HTTP 调用已经加载的模型。
- CFPRF 权重仅在常驻模型进程启动时加载一次；串行推理受互斥锁保护。`/health` 返回 `pid`、`model_load_count` 和 `requests_completed`，用于验证复用。
- 接受 WAV/FLAC/MP3、单/双声道；服务有效时长为 **0.5–21.25 秒**。超长输入返回 `DURATION_OUT_OF_RANGE`，不静默裁剪。
- 定位保留原 PRN 区间分数，允许 `timeline=null`，不人工构造时间线。分数未经概率校准；整段分数是保留区间分数最大值。

模型版本由 `services/configure_cfprf.py` 根据权重 SHA-256、模型及适配器源码、输入和判定规则生成；常驻进程启动时核对实际文件哈希。代码或权重改变后执行：

```bash
python services/configure_cfprf.py
```

然后重新启动服务。已保存任务仍保留受理时的版本。真实配置位于 `model_integration/cfprf-resident.json`；命令行备用配置也已填写到 `cfprf-command.json`。

独立运行 `python run.py` 只启动后端，需要先启动模型服务；推荐使用外层 `start.sh` 管理两者。原 `.env.example` 保留模拟模式供独立开发测试，当前实际使用的 `.env` 已关闭模拟。

## 验证记录

本机已经通过 30 条随包音频的 HTTP 上传、真实模型推理、结果查询及预览接口；另验证检测模式、48 kHz 双声道转换和 22 秒输入拒绝。连续任务模型 PID 不变、加载计数为 1。报告位于外层 `methods/cfprf/results/fullstack_validation.json`，完整结果示例为 `fullstack_example.json`。

这些检查验证工程链路，不将所有任务成功理解为检测正确率。模型仍采用 HAD 官方权重，未用这 30 条音频调阈值或训练。
