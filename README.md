# AlohaMini Platform

**English** | [简体中文](README.zh-CN.md)

Control, data collection, and algorithm development for AlohaMini dual-arm mobile robots. A Raspberry Pi runs the Host to connect the hardware; the PC handles teleoperation, training, evaluation, and ROS 2 development.

## Documentation

[Browse all guides](docs/README.md)

- [Installation](docs/install.md): PC, Raspberry Pi, policy extensions, and simulation environments.
- [AlohaMini user guide](docs/alohamini.md): models, wiring, calibration, Host startup, teleoperation, and local data collection.
- [Client API reference](docs/host-protocol.md): Python APIs and communication contracts for custom clients and policies.
- [ROS 2](docs/ros2.md): robot state, cameras, motion control, and MoveIt.
- [EDU navigation](docs/ros2-navigation.md): Lidar, IMU, mapping, and autonomous navigation.
- [Simulation](docs/simulation.md): viewing, scenes, and replay with MuJoCo, ManiSkill, and RoboTwin.
- [LeRobot](docs/lerobot.md): dataset and model compatibility.
- [Training and deployment](docs/training.md): dataset selection and conversion, policy training, offline checks, remote inference, and hardware evaluation.
- [Notebooks](docs/notebook.md): optional data analysis and experiment tools.

## Daily startup

After installation, device configuration, and calibration, run on the Raspberry Pi:

> Startup enables servo torque and homes the lift against its lower stop. Support both arms and clear the lift descent path first. Support any parts that may fall before stopping or disconnecting power.

```bash
conda activate alohamini_host
alohamini host --robot_model alohamini2pro
```

The forward and right wrist cameras are enabled by default. With no cameras connected, append `--cameras` without any names.

On the PC, replace `<HOST_PI>` with the Raspberry Pi address and use the same model as the Host:

```bash
conda activate alohamini
alohamini inspect --host <HOST_PI> --model alohamini2pro
```

`inspect` only reads state and sends no motion commands. Working files default to `~/Alohamini_workspace/`; run `alohamini paths` to inspect the paths.
