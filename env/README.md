# 依赖维护

项目依赖在 `pyproject.toml` 声明。`dependencies.toml` 选择 PC 所需 extras、部署版本和更新分组；`.in`、pip `.lock` 文件由脚本生成。Conda 的 Python、FFmpeg 和基础工具版本仍通过 `environment.yml`、`host.yml` 及对应 explicit 文件维护。

在 Conda `alohamini` 环境中运行，先安装 `dependencies.toml` 指定版本的 `uv`：

```bash
python env/dependencies.py check
python env/dependencies.py refresh
python env/dependencies.py refresh --group models --report /tmp/dependency-changes.md
```

`refresh` 只修改依赖文件。已有锁定版本优先保留；指定分组时，检查同一主版本中发布满 7 天的稳定版本并重新解析。解析失败会恢复原文件。`models`、`compute`、`host`、`tools` 分别维护模型、计算、硬件与其余工具/数据依赖。PC/Host 的公共依赖保持相同版本。

自动更新的主版本上限也约束间接依赖。跨主版本升级时，先显式修改项目声明或部署约束，再运行不带 `--group` 的 `refresh` 并执行回归。

## GitHub Actions

- `Dependency validation`：相关 PR、main 提交及手动触发；检查清单，在新建 Conda 环境中安装 CPU 对应版本，运行算法、checkpoint、续训、精度和分布式验证回归。CPU 锁定文件仅作为 CI 产物，不替换 CUDA 部署锁。
- `Dependency update candidates`：每周一 03:17 UTC，也可手动选择分组；生成草稿 PR 并显式触发候选提交的 CPU 检查。PR 记录版本差异，运行产物保留 30 天。
- `Dependency GPU validation`：手动选择候选分支，填写空闲 GPU 编号和 checkpoint 用例文件。需要标签为 `alohamini-gpu` 的 Linux x86_64 自托管 runner，提供 Conda、兼容驱动及至少两张空闲 GPU。用例路径上的 checkpoint、基座资源和数据集必须可读。

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
