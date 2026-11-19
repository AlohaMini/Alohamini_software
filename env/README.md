# 依赖维护

项目依赖在 `pyproject.toml` 声明。`dependencies.toml` 选择 PC 所需 extras、部署版本和更新分组；`.in`、pip `.lock` 文件由脚本生成。Conda 的 Python、FFmpeg 和基础工具版本仍通过 `environment.yml`、`host.yml` 及对应 explicit 文件维护。

在 Conda `alohamini` 环境中运行，先安装 `dependencies.toml` 指定版本的 `uv`：

```bash
python env/dependencies.py check
python env/dependencies.py refresh
python env/dependencies.py refresh --group models --report /tmp/dependency-changes.md
```

`refresh` 只修改依赖文件。已有锁定版本优先保留；指定分组时，检查同一主版本中发布满 7 天的稳定版本并重新解析。解析失败会恢复原文件。`models`、`compute`、`host`、`tools` 分别维护模型、计算、硬件与其余工具/数据依赖。PC/Host 的公共依赖保持相同版本。

指定分组只更新该组及属于该组的间接依赖，其他组保持锁定；不自动降级或跨主版本。需要联动迁移多个组时，先显式修改项目声明或部署约束，再运行不带 `--group` 的 `refresh` 并执行回归。

macOS ARM64 锁随 `refresh` 更新；单独重新解析使用 `python env/dependencies.py mac-lock`。

## GitHub Actions

- `macOS native compatibility`：Mac 分支、main 和相关 PR 运行；Apple Silicon、Intel 串口检查与 PC 检查并行运行；PC 分为 `core`（通信、缓存、数据与视频）和 `training`（训练与续训）两组。pip 下载缓存按架构与依赖锁区分；每次仍执行锁定安装和依赖检查。各组分别上传日志、JSON 与 JUnit 报告。MPS 和 VideoToolbox 硬件测试在能力可用时执行。
- `Dependency validation`：相关 PR、main 提交及手动触发；检查清单，在新建 Conda 环境中安装 CPU 对应版本，运行算法、checkpoint、续训、精度和分布式验证回归；另在 x86_64、ARM64 runner 安装 Host 锁并检查串口库、图像及进程内通信。CPU 锁定文件仅作为 CI 产物，不替换 CUDA 部署锁。
- `Dependency update candidates`：每季度首日（1、4、7、10 月 1 日）03:17 UTC，也可手动选择分组；只运行所选组，生成草稿 PR 并触发候选提交的 CPU、Host 检查。候选生成成功不代表兼容检查通过，检查入口见运行摘要，测试摘要列出失败用例。PR 记录版本差异，运行产物保留 30 天。
- Dependabot：每季度检查 GitHub Actions 版本，同组更新合并为一个 PR。
- `Dependency GPU validation`：手动选择候选分支，填写空闲 GPU 编号和 checkpoint 用例文件。需要标签为 `alohamini-gpu` 的 Linux x86_64 自托管 runner，提供 Conda、兼容驱动及至少两张空闲 GPU。用例路径上的 checkpoint、基座资源和数据集必须可读。

大模型权重超过单卡显存时，将 `checkpoint_backend` 设为 `fsdp2`，使用所选显卡中的前两张比较推理结果，保留 checkpoint 原有权重精度。

仓库 Actions 设置须允许创建 PR；组织设置也须允许此权限。版本候选保持草稿，不自动合并。模型/计算更新接受前，需 CPU 与 GPU 检查均通过。GPU 工作流使用独立的临时环境，不更新现有运行环境；若指定显卡有计算进程则退出。

## 旧 checkpoint 对比

复制 `checkpoint-cases.example.json` 到运行产物目录，填写真实 checkpoint、数据集和代表性样本。`indices` 是所选回合组成的数据集中的样本索引；使用 checkpoint 自带动作单位，按任务设定 `atol`、`rtol`。视觉目标策略需另外填写 `goal_images`。

分别在原环境与候选环境捕获，再比较：

```bash
python env/checkpoint_probe.py capture --cases /path/to/cases.json --device cuda:0 --output /tmp/baseline.json
python env/checkpoint_probe.py capture --cases /path/to/cases.json --device cuda:0 --output /tmp/candidate.json
python env/checkpoint_probe.py compare --reference /tmp/baseline.json --candidate /tmp/candidate.json --output /tmp/comparison.json
```

对比要求 checkpoint、配置、输入与随机种子相同；记录动作差异、推理耗时、显存及依赖版本。输出不在容差内时返回失败。性能记录用于审查，短样本耗时不作为自动接受阈值。

计算库升级允许数值变化时，可在用例中设置 `"acceptance": {"metric": "mae", "rtol": 0.02, "atol": [...]}`。`atol` 必须按动作维度顺序填写各自单位下的绝对误差预算；每维新版 MAE 须不超过 `旧版 MAE × (1 + rtol) + atol`。使用预先选定的验证样本与阈值，排除尾部补位及历史动作，仍保留输出差异。该检查衡量离线动作误差，不替代闭环任务评估。
