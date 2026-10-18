# 工程交付验证记录

验证日期：2026-10-13。所有检查均针对本次前后端工程，验证软件流程与接口一致性，不评估真实伪造检测准确率。

## 实际执行结果

| 检查 | 结果 |
| --- | --- |
| 后端自动化测试 | 27 项通过，包含真实 FFmpeg 解码、任务子进程、模型输出校验、预览、失败、过期、重启、超时及队列限制 |
| 模型接入自动化测试 | 13 项通过，包含 state_dict 实际加载、SHA-256、分数方向、重采样、时间映射、严格配置以及通过 HTTP 服务调用检查点 |
| 输出适配示例自动化测试 | 19 项通过，包含元组输出、分数方向、独立整段输出、错误形状、非有限值、最后不足一格的覆盖、严格权重加载及禁止覆盖已有文件 |
| 合并 Python 测试 | 59 项全部通过 |
| 前端 JavaScript 逻辑测试 | 14 项全部通过，涵盖原始分数、三种定位状态、模型来源、异步查询、请求失败、重试和停止等待 |
| 30 条带标签音频 | 清单、详细 JSON、CSV、音频与标注校验值、区间边界及覆盖一致；真实、整段伪造、局部伪造各 10 条 |
| FLAC 与 MP3 上传 | 两种格式均经过实际转码、HTTP 上传和 FFmpeg 解码，任务成功，预览时长与结果一致 |
| 30 条样本 HTTP 全流程 | 全部成功上传，异步任务成功；返回符合 OpenAPI，完整时间序列覆盖音频，WAV 预览时长与原文件一致；使用明确标识的模拟适配器 |
| 浏览器连接模拟服务 | 上传、任务查询、原始分数、时间标红、片段试听、预览、JSON 导出、仅检测状态、历史及移动菜单通过；无未处理 JavaScript 错误 |
| 浏览器连接权重适配器 | 实际加载模型源码及 `.pt` 检查点，执行音频推理，展示负分数和空定位结果、导出完整来源；无未处理 JavaScript 错误 |
| 输出适配示例完整流程 | 单音频检查、3 条样本 HTTP 契约与预览校验、浏览器上传及定位试听、JSON 导出、仅检测、历史和移动宽度检查通过；仍为 `is_mock=true` |
| 前端保持不变 | 全部 10 个前端文件与上一交付 ZIP 逐字节一致；后端业务代码及接口契约同样保持不变 |

浏览器检查使用桌面 1440×1050 与移动宽度 390 的页面，移动页面无水平溢出，菜单可以展开和关闭。通过浏览器返回的报告保留任务 ID、模型版本、`is_mock`、原始分数与时间区间。

权重加载使用随包 `make_fixture.py` 生成的简单工程测试模型，始终标记为 `is_mock=true`。该模型没有训练为伪造检测器；实际执行过神经网络参数加载和计算，不代表已验证任何候选研究模型或真实检测效果。未对 GPU 推理速度、不同 CUDA 环境、Windows/macOS 或公网部署作运行验证。

新增输出适配示例使用 `make_output_adapter_example.py` 生成的独立 CPU 测试模型。它将元组及 logits 输出转换为既有接口；实际加载了包装层权重并执行推理，页面全流程无需改动前端。两种工程测试模型均不构成真实检测效果验证。

## 实测环境

Linux，Python 3.13.5，FastAPI 0.135.1，Uvicorn 0.41.0，python-multipart 0.0.22，NumPy 2.1.3，PyYAML 6.0.2，HTTPX 0.28.1，pytest 8.3.4，jsonschema 4.23.0，PyTorch 2.7.1+cu118（本次推理用 CPU），SciPy 1.15.3，FFmpeg 4.2.7。

前端逻辑验证使用 Node.js 20.18.0；浏览器验证使用 Playwright 1.48.0 和其 Chromium 130 测试环境。Node.js 和浏览器自动化工具只用于开发验证，不是网页或后端运行依赖。

## 复现命令

在工程根目录、已安装 FFmpeg 的独立 Python 环境执行：

```bash
python -m pip install -r backend/requirements-dev.txt
# 按实际设备安装匹配的 PyTorch；模型测试还需通用桥接依赖。
python -m pip install -r model_integration/requirements-model.txt
python -m pytest -q
node frontend/tests/analysis.test.js
node frontend/tests/api.test.js
python tools/validate_samples.py
python tools/verify_package.py
```

启动默认模拟服务后，在另一终端执行：

```bash
python tools/smoke_samples.py --base-url http://127.0.0.1:8000
```

若需要自动浏览器验收，另行安装与操作系统兼容的 Playwright 和 Chromium：

```bash
python -m pip install playwright
python -m playwright install chromium
python tools/smoke_browser.py --base-url http://127.0.0.1:8000
```

该浏览器脚本针对模拟模型的工程验收，默认不接受真实模型作为测试替代。需要验证实际权重加载时，先运行 `model_integration.make_fixture`，按模型接入说明将服务指向生成配置，再为浏览器脚本传入 `--model-id example-plumbing-fixture`。模拟服务返回成功只说明工程流程可用，真实检测验收还需要模型提供方的参考输出与独立数据。

输出适配示例按 [完整操作说明](model-adapter-walkthrough.md) 生成并启动后，可用现有验收工具验证：

```bash
python tools/smoke_samples.py --base-url http://127.0.0.1:8000 --model-id example-output-adapter --limit 3
python tools/smoke_browser.py --base-url http://127.0.0.1:8000 --model-id example-output-adapter
```

## 交付完整性

`SHA256SUMS.txt` 为包内文件校验表；`.env`、运行数据、缓存、下载的浏览器、现有服务器环境及外部权重均不纳入工程包。压缩包外另附 ZIP 的 SHA-256 文件。解压后可用上述 `verify_package.py` 检查文件完整性。
