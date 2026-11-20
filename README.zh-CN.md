# AlohaMini Platform

[English](README.md) | **简体中文**

AlohaMini 双臂移动机器人的控制、数据采集与算法开发平台。树莓派运行 Host 连接硬件，PC 进行遥操、训练、评估及 ROS2 开发。

## 文档

[文档目录](docs/zh-CN/README.md)

- [安装](docs/zh-CN/install.md)：PC、树莓派、策略扩展与仿真环境。
- [AlohaMini 使用手册](docs/zh-CN/alohamini.md)：型号、接线、标定、Host、遥操与本地数采。
- [客户端接口参考](docs/zh-CN/host-protocol.md)：供自定义客户端与策略开发使用的 Python API 和通信约定。
- [ROS2](docs/zh-CN/ros2.md)：整机状态、相机、运动控制与 MoveIt。
- [EDU 导航](docs/zh-CN/ros2-navigation.md)：雷达、IMU、建图与自主导航。
- [仿真](docs/zh-CN/simulation.md)：MuJoCo、ManiSkill、RoboTwin 的查看、场景与回放命令。
- [LeRobot](docs/zh-CN/lerobot.md)：数据与模型兼容范围。
- [训练与部署](docs/zh-CN/training.md)：数据选择与转换、策略训练、离线检查、云端推理与真机评估。
- [Notebook](docs/zh-CN/notebook.md)：可选的数据分析与实验入口。

## 日常启动

完成安装、设备配置与标定后，在树莓派执行：

> 启动会使能舵机并执行升降触底回零。先支撑双臂，清空升降下降路径；停止或断电前也应支撑可能下落的部件。

```bash
conda activate alohamini_host
alohamini host --robot_model alohamini2pro
```

默认开启前视和右腕相机；没有相机时，在命令末尾加 `--cameras`，不填名称。

在 PC 执行，将 `<HOST_PI>` 换成树莓派地址，型号与 Host 保持一致：

```bash
conda activate alohamini
alohamini inspect --host <HOST_PI> --model alohamini2pro
```

`inspect` 只读状态，不发送运动命令。工作文件默认放在 `~/Alohamini_workspace/`；执行 `alohamini paths` 查看路径。
