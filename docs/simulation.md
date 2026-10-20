# MuJoCo

支持 AlohaMini2Pro 固定底盘的交互查看、模型检查、完整位姿 IK 和动力学关节轨迹执行。底盘三个虚拟关节及三个轮轴固定；双臂、夹爪和升降轴保留重力、关节限位与电机力限制。

## 安装

在仓库根目录执行：

```bash
conda activate alohamini
python -m pip install -e ./simulation/mujoco
```

## 交互查看

在桌面会话中从仓库根目录执行，无需输出目录：

```bash
MUJOCO_GL=glfw alohamini-mujoco view \
  --initial-positions '{"vertical_move":0.2}' \
  --trajectory examples/simulation/follow.json
```

不传 `--trajectory` 时通过动力学保持初态；`--paused` 暂停启动，`--collisions` 直接显示物理凸包。鼠标左键拖动旋转，右键拖动平移，滚轮缩放。

- 空格：播放或暂停；轨迹结束后自动暂停。
- F8：切换完整视觉网格与实际接触计算使用的碰撞凸包。
- F9：复位到配置初态并暂停；再次按空格重播。
- 右方向键：暂停时推进一个物理步。
- F6：切换坐标系显示；关闭窗口退出。

窗口显示仿真时间、实际运行倍率、接触数和电机力饱和步数。播放保留完整物理步，复杂碰撞候选可能低于实时速度。`--collision-profile` 和 `--collision-bundle` 与模型检查相同。原生面板用于查看，初态与物理配置由启动参数指定；面板编辑不会改变动力学播放。当前窗口包含机器人，尚无任务桌面或物体；查看与播放不代替接触验收。

## 模型检查

输出目录必须不存在：

```bash
MUJOCO_GL=egl alohamini-mujoco inspect \
  --initial-positions '{"vertical_move":0.2}' \
  --render --output ~/Alohamini_workspace/reports/simulation/asset
```

输出 `model.xml`、网格、来源校验值、运动学/惯量检查、夹爪开口截面和两张几何预览。导出的模型目录可整体移动。`--compare-maniskill /路径/ManiSkill` 对照本地 AlohaMini2Pro 描述和网格。

`--collision-profile authoritative` 保留权威 URDF 的碰撞体。`cad-pieces` 使用原始分解的指部凸块作比较；两者均保留掌部、相机和连杆碰撞体，均未完成真实接触标定。截面坐标属于各侧 `Fixed_Jaw`，开口沿局部 X 测量。

## CAD 碰撞候选

```bash
python -m pip install -e './simulation/mujoco[geometry]'
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 alohamini-mujoco decompose \
  --links left_Moving_Jaw right_Moving_Jaw --tolerance-m 0.0005 --seed 0 \
  --output ~/Alohamini_workspace/reports/simulation/moving-jaw
alohamini-mujoco inspect --initial-positions '{"vertical_move":0.2}' \
  --collision-bundle ~/Alohamini_workspace/reports/simulation/moving-jaw \
  --output ~/Alohamini_workspace/reports/simulation/moving-jaw-inspection
```

每个凸体独立保存，使用 URDF 连杆局部坐标和米制单位；来源 CAD、局部变换、分解参数和校验值写入 `collision.json`。加载时验证来源、坐标、校验值及凸体闭合性。只替换清单中指定连杆的碰撞体，完整视觉网格、惯量和 SRDF 过滤保持有效。`follow` 同样支持 `--collision-bundle`。

分解采用 CoACD，不限制凸体数量，不做减面或膨胀。MSH 使用 float32，在保存后的坐标上重建凸包边界三角形，记录坐标量化误差。`--tolerance-m` 是算法的凹度目标，不代表表面误差上界。精度缝合位移最多 50 nm；非流形 CAD 默认拒绝，`--nonmanifold remesh` 显式生成重网格候选，须另行验证几何变化。当前固定夹爪 CAD 含非流形部分，默认预检会拒绝。

`inspect` 的 `convexification` 报告列出每个物理网格的凸包与源网格诊断；非闭合或朝向不一致的网格不提供体积比。夹爪截面报告两指各自接触面的 X 坐标、偏差和命中统计，并比较开口宽度，避免两指同向偏移时误差抵消。`probe_counts` 区分开口配对是否可测量，缺失不等于零误差。`validation.passed` 仅表示模型导入一致性通过，候选仍须验证接触、间隙与动力学性能。碰撞候选不会自动成为默认模型。

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
