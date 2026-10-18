# Git 维护说明

仓库用于维护模型封装、路由、前后端、测试及文档。权重、运行环境、下载目录、音频和推理产物保留在本地，不随 Git 提交。

## 首次提交

在项目根目录执行，先确认当前目录尚未初始化为有效仓库：

```bash
git init -b main
git status --short --untracked-files=all
git add .
git diff --cached --stat
git diff --cached --check
git diff --cached
git commit -m "chore: initialize audio forgery detection project"
```

提交前检查暂存区，确认没有凭据、个人路径、上传文件或模型权重。Git 提交会记录作者姓名和邮箱；若不希望公开个人邮箱，可在本仓库使用 GitHub 账户设置中提供的 noreply 邮箱。不要填写他人的身份。

在 GitHub 创建空仓库后，使用自己的仓库地址配置远端并推送：

```bash
git remote add origin <你的GitHub仓库地址>
git push -u origin main
```

## 提交范围

- 保留 `methods/`、`services/`、`voxguard-fullstack/` 中的源码、测试、文档和通用配置示例。
- 保留 `third_party/` 中现有框架源码及其版权、许可文件；`temp/` 中下载的上游仓库不提交。
- 不提交 `.envs/`、`.cache/`、`temp/`、音频、权重、`results—temp/`、`methods/*/results/` 和前后端运行数据。
- 不提交生成的 `cfprf-command.json`、`cfprf-resident.json`、`ensemble-resident.json`；通用 `*.example.json` 和 `.env.example` 保留。
- 第三方作者署名、来源和许可说明应保留；本次未为整个项目自行指定开源许可证。

`.gitignore` 不会删除本地文件，也不会自动停止跟踪已提交的文件。不要使用 `git add -f` 绕过规则提交本机文件。

## 新机器恢复

克隆仓库不会带上模型权重、`temp/` 上游代码、音频或 Python 环境，需要按各方法 README 准备依赖和权重，并恢复对应上游源码。当前仓库并非仅克隆即可直接运行的完整发布包。

准备好模型环境与权重后，根据需要生成部署配置：

```bash
python services/configure_ensemble.py
# 单独部署 CFPRF 时：
python services/configure_cfprf.py
```

环境变量按 `voxguard-fullstack/.env.example` 和实际部署方式配置，五模型接入见 [常驻路由说明](voxguard-fullstack/docs/ensemble-resident.md)。

`voxguard-fullstack/SHA256SUMS.txt` 是原完整工程包的校验表，包含部分不会提交的部署配置和音频。Git 克隆目录不等同于原工程包，缺少这些文件时不能直接通过完整包校验；发布完整工程包时应重新生成与实际打包内容一致的校验表。

## 日常修改

```bash
git switch -c feat/your-change
git status --short
git diff
# 按修改范围运行相应测试，再提交具体文件。
git add <文件路径>
git diff --cached
git commit -m "feat: describe the change"
```

检查单个文件的忽略原因：

```bash
git check-ignore -v temp/weights/example.pth
```
