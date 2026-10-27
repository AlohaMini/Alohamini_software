# 开发

核心协议解析只依赖 Python 标准库。网络客户端额外使用 pyzmq；模块导入不会启动网络连接或加载机器人硬件。

开发使用[安装指南](docs/install.md)中的 `alohamini` 环境。Python 源码保持 3.10 兼容，主开发环境使用 3.12.13；调整公共接口时应覆盖这两个版本。

## 目录

```text
Alohamini/
├── environment.yml          # Conda 直接依赖版本
├── .python-version          # 主开发解释器版本
├── env/                     # 按架构锁定的 Conda / pip 依赖
├── pyproject.toml           # 核心包、入口与检查工具配置
├── src/alohamini/
│   ├── cli.py               # 命令行入口
│   ├── errors.py            # 公共异常
│   ├── client.py            # 共用机器人客户端
│   ├── protocol.py          # 已部署 Host 协议编解码
│   ├── model/               # 型号定义与资产加载
│   ├── hardware/            # 硬件读写
│   ├── runtime/             # Host 控制与保护
│   ├── calibration/         # 标定与单位转换
│   ├── datasets/            # 本地数据存储、检查与格式导出
│   └── apps/                # 遥操、数采与可视化
├── tests/                   # 核心测试与跨组件测试
└── docs/                    # 安装运维与接口文档
```

核心运行代码放在 `src/alohamini/`；命令行入口只负责参数与输出，不复制协议逻辑，框架依赖不得进入通用接口。测试辅助代码不随运行包分发。

`apps/` 调用客户端和数据接口组织操作流程；`datasets/` 不依赖应用入口。Host、硬件和协议不反向导入应用或数据处理代码。仅进行文件格式转换的导出器属于 `datasets/`；需要学习框架运行时的适配代码属于 `integrations/`。

旧实现留在原仓库和 Git 历史中。迁入功能成为正式实现，不另建历史副本。兼容层必须有实际接入需求，只保留必要的接口适配；调用方迁移完成后删除，不能持续容纳旧业务逻辑。

ROS2、学习框架与仿真扩展分别归属同级 `ros2/`、`integrations/` 和 `simulation/`，随实际功能建立。各扩展管理自身依赖与测试，不放入核心包。

个人开发计划与进度存放于 Obsidian，不放在源码仓库。通用型号资产以根目录 `models/` 为权威来源；单机标定与型号资产分开保存。

`docs/` 面向安装、使用与接口查询，不记录 AI 工作过程、源码核对流水或测试历史。
个人训练配置、数据、标定及运行产物放在 `~/Alohamini_workspace/`；仓库只保留通用示例和型号资源。
提交 Notebook 前清空运行输出与执行计数。临时本地文件可放在已忽略的 `.local/` 或 `configs/`，不纳入发行包。

## 分支与工作区

`main` 保存已验证的通用功能，`~/Alohamini` 是唯一源码开发与测试目录，日常查看和使用 `main`。一个任务使用一个分支，在该目录中切换；明确指定算法分支的任务在对应分支完成。除非用户明确要求，不创建额外 worktree，也不在 `~/Alohamini_workspace/development/` 或其他外部目录开发。`~/Alohamini_workspace/` 用于数据、训练配置、日志、运行产物和备份。

- `feat/am-act-conditional-cvae`：AM-ACT 算法及必要接口适配。
- `feat/diffusion-algorithms`：扩散算法及必要接口适配。
- 仿真、精度调试和其他通用功能按主题独立开发，通过影响范围内的检查后进入 `main`，不长期混入某个算法分支。

通用功能和修复从最新 `main` 开始，在短分支完成；合入主分支后，通过合并 `main` 同步到所有仍在使用的算法和功能分支。开始算法开发前先同步 `main`，保留算法自身的实现与配置，不长期维持缺少通用能力的旧基线。尚未进入主分支的新算法，其内部修复随算法分支开发。归档和备份分支保留原始基线。

开始任务前检查当前分支、上游、工作区改动和已有 worktree。共享目录有其他任务写入时，先保留并协调现有工作；不要通过切换、reset、stash 或清理来腾出目录。多个会话串行使用该目录，不以新增外部 worktree 绕过占用。

cherry-pick 用于明确的选择性移植，不作为算法分支长期同步主分支的默认方式。判断修复是否已发布时，核对 `origin/main`、实际内容及补丁等价关系，不能只比较提交哈希或 ahead/behind 数量；保留既有历史，不为减少提交数量重写已共享分支。

既有外部 worktree 迁回前核对分支、未提交内容及本地资源；保留独有提交与文件后再注销工作树。仓库内的 `development/` 存放开发工具，不是临时 worktree 目录，不能按缓存清理。

## 检查

在仓库根目录运行离线测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

未安装 pyzmq 时跳过本机网络测试。网络测试只绑定 `127.0.0.1` 的临时端口，不连接机器人。具体测试可使用 `unittest discover -p` 选择。

迁移差分检查可通过 `ALOHAMINI_SOURCE_REPO`、`ALOHAMINI_ROS_SOURCE` 指向可信的旧仓库，再运行 `unittest discover -s tests -p test_migration.py`。检查从指定工作树读取原函数进行比较，不加载旧硬件入口；未设置路径时跳过这组检查。

可选代码检查：

```bash
ruff check src tests
ruff format --check src tests
```

新增行为应覆盖对应输入、输出和失败边界。公共接口说明随行为变化更新；机器人实测应明确设备、软件版本及测试范围，不能由离线测试结果替代。
