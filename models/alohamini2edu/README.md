# EDU 模型

机身、双臂和升降使用当前 `alohamini2pro` 的几何、碰撞体与关节限位。C1 雷达使用 56 × 56 × 42 mm 的包围盒，尺寸及扫描平面到盒中心的偏移见 `config/lidar_collision.json`；安装位姿见 `config/sensors.yaml`。

更新 Pro 模型或雷达名义安装位姿后，在仓库根目录同步并检查：

```bash
conda activate alohamini
python scripts/sync_edu_model.py
python -m pytest tests/test_edu_model.py -q
```

随后重新构建 ROS 工作区。设备 Home 标定与传感器实测外参保存在各机器的工作目录中。
