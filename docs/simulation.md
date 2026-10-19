# MuJoCo

支持 AlohaMini2Pro 固定底盘的模型检查、完整位姿 IK 和动力学关节轨迹执行。底盘三个虚拟关节及三个轮轴固定；双臂、夹爪和升降轴保留重力、关节限位与电机力限制。

## 安装

在仓库根目录执行：

```bash
conda activate alohamini
python -m pip install -e ./simulation/mujoco
```

## 模型检查

输出目录必须不存在：

```bash
MUJOCO_GL=egl alohamini-mujoco inspect \
  --initial-positions '{"vertical_move":0.2}' \
  --render --output ~/Alohamini_workspace/reports/simulation/asset
```

输出 `model.xml`、网格、来源校验值、运动学/惯量检查、夹爪开口截面和两张几何预览。导出的模型目录可整体移动。`--compare-maniskill /路径/ManiSkill` 对照本地 AlohaMini2Pro 描述和网格。

`--collision-profile authoritative` 保留权威 URDF 的碰撞体。`cad-pieces` 使用原始分解的指部凸块作比较；两者均保留掌部、相机和连杆碰撞体，均未完成真实接触标定。截面坐标属于各侧 `Fixed_Jaw`，开口沿局部 X 测量。

## 轨迹执行

```bash
alohamini-mujoco follow examples/simulation/follow.json \
  --initial-positions '{"vertical_move":0.2}' \
  --output ~/Alohamini_workspace/reports/simulation/follow
```

轨迹 JSON 声明 `schema_version: 1`、`joint_names`、`times_s` 和二维 `positions`。时间从零开始且严格递增；位置使用 URDF 的 rad/m，首个路点须匹配初态。三次插值的峰值速度必须符合 URDF 限制。未指定的关节保持初始位置。

执行通过力受限 PD 和重力/科氏力前馈推进物理步，不直接写入关节位置。报告包含跟踪误差、实际速度、力饱和、关节越限和接触深度；验收失败返回非零退出码。

接口约定：`T_A_B` 将 B 中坐标转换到 A；四元数为 `wxyz`。`solve_ik` 在独立状态上求解世界坐标下的 TCP 完整位姿，返回接触诊断；IK 收敛不代表整条路径可避障。

预览相机用于检查资产；机器人镜头内外参、任务物体接触参数和真机执行器参数须分别标定。此接口不执行真机命令。
