# 运动学派生字段

原始记录保留 Host 字段、动作请求和时间戳。数据层提供两个可组合的转换：

- `KinematicFields`：命名关节位置与夹爪 → 绝对 TCP 位姿、夹爪及透传字段。
- `RelativeActions`：绝对目标 ↔ 相对同一观测锚点的目标；线性量相减，旋转在 SO(3) 上组合。

两者不依赖 Torch、ROS 或策略模型，可供数据导出和训练 processor 调用。几何计算位于
`alohamini.kinematics.arm`；ROS 的周期跟踪增益、速度约束仍由 ROS 遥操实现负责。

## 转换描述

安装独立依赖：`pip install -e '.[kinematics]'`。

由数据集的 Host 单位与**该数据对应的已安装关节标定**生成几何快照：

```bash
python -m alohamini.datasets.kinematic_fields \
  --dataset /path/to/dataset \
  --left-map /path/to/hardware_joint_map_left.yaml \
  --right-map /path/to/hardware_joint_map_right.yaml \
  --output /path/to/kinematic_fields.json
```

描述包含源字段顺序、编码器零位/方向/范围、DH 与 TCP 变换、每臂参考坐标系、关节限位、
夹爪映射和透传字段单位。缺少物理标定的归一化位置不能自动当作弧度。
夹爪默认通过明确的开闭端点映射到 `canonical_0_0.1`，该数值不声明为实际开口米数。
每臂 TCP 使用对应的 `left_Base`、`right_Base`，不隐含世界坐标系或移动底盘里程计。

```python
import json
from alohamini.datasets.kinematic_fields import KinematicFields

with open("kinematic_fields.json") as stream:
    fields = KinematicFields(json.load(stream))
derived = fields.derive(row, {
    "observation.state": "observation.tcp_state",
    "action": "tcp_action",
})
feature = fields.feature()  # 名称、单位、坐标系与可保存的转换描述
```

`derive` 新增字段，拒绝覆盖原字段；向量和时间窗口均可转换。该 API 不重采样时间戳，
也不把动作请求改成未来反馈。只有关节数据的来源语义保持一致，派生位姿才可作为同类标签。

## 相对目标

```python
from alohamini.datasets.action_transforms import RelativeActions

codec = RelativeActions(
    width=18,
    linear_dims=[0, 1, 2, 7, 8, 9],
    rotation_groups=[[3, 4, 5], [10, 11, 12]],
    reference="anchor",
)
delta = codec.encode(current_tcp_state, future_tcp_targets)
absolute = codec.decode(current_tcp_state, delta)
```

每个目标都相对同一个 `current_tcp_state`：`Δp = p_target - p_anchor`，
`ΔR = R_target @ R_anchor.T`。还原时 `R_target = ΔR @ R_anchor`。
旋转向量不直接相减，目标不逐帧累积；未选择的夹爪、底盘速度和升降高度保持绝对值。

部署 IK 使用对应观测关节作为初值，后续目标延续上一步解。
不可达目标、超出初值限位或不满足指定误差的解会报错，不返回未收敛的关节目标。
