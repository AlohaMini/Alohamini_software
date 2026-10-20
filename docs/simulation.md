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

窗口显示仿真时间、实际运行倍率、接触数和电机力饱和步数。播放保留完整物理步，复杂碰撞候选可能低于实时速度。`--collision-profile` 和 `--collision-bundle` 与模型检查相同。原生面板用于查看，初态与物理配置由启动参数指定；面板编辑不会改变动力学播放。未指定场景时仅包含机器人；查看与播放不代替接触验收。

只查看指定连杆：

```bash
alohamini-mujoco view --collisions --paused \
  --initial-positions '{"vertical_move":0.2}' \
  --links vertical_link left_Upper_Arm right_Upper_Arm
```

## LIBERO Spatial 场景

导入使用独立的 LIBERO Python 环境，运行场景仍使用 `alohamini`。来源须为干净的 LIBERO 提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`；导入环境须有 MuJoCo 3.8.1、robosuite 1.4.0 及 LIBERO 依赖。导入不会修改该环境或全局 LIBERO 配置。输出目录不得存在：

```bash
alohamini-mujoco libero-import \
  --source ~/Alohamini_workspace/sources/LIBERO \
  --python ~/miniconda3/envs/alohamini_sim/bin/python \
  --output ~/Alohamini_workspace/simulation/libero_spatial_8f1084e_v2
MUJOCO_GL=glfw alohamini-mujoco view \
  --libero-bundle ~/Alohamini_workspace/simulation/libero_spatial_8f1084e_v2 \
  --scene-config examples/simulation/libero_spatial.json \
  --task-index 0 --reset-index 0
```

支持 10 条 Spatial 任务，各保留 50 个具名物体初态；不迁移 Panda 关节或 7 维动作。快照包含原物体网格、纹理、分块碰撞体、惯量、接触参数、BDDL 和文件校验值。`--task-index` 为 0–9，`--reset-index` 为 0–49。其他 LIBERO 套件和谓词尚不支持。

场景配置明确 `T_world_root`、`T_world_libero`、机器人初态、支撑台和相机。LIBERO 变换须保持竖直轴，不缩放任务物体。示例中桌面高 0.5 m，机器人固定在支撑台上；相机是未完成真机标定的仿真配置。该布局须结合碰撞资产验证固定体间隙和任务可达性。场景检查的 `passed` 表示导入等价；`static_robot_scene.passed` 单独检查 MuJoCo 接触列表会跳过的固定体重叠。

`inspect`、`follow` 和 `collect` 使用同样的场景参数；`inspect` 的整体结果同时要求固定体检查通过。当前原 URDF 底座整体凸包与示例桌面重叠，须先选择并验收合适的碰撞候选。此适配使用独立 benchmark 名称，不产生原 Panda LIBERO 可比得分。

## 任务数采与回放

```bash
python -m pip install -e './simulation/mujoco[collection]'
MUJOCO_GL=egl alohamini-mujoco collect "${TRAJECTORY_JSON:?请设置任务轨迹 JSON 路径}" \
  --libero-bundle ~/Alohamini_workspace/simulation/libero_spatial_8f1084e_v2 \
  --scene-config examples/simulation/libero_spatial.json \
  --task-index 0 --reset-index 0 \
  --collision-bundle "${COLLISION_BUNDLE:?请设置已检查的碰撞候选目录}" \
  --output ~/Alohamini_workspace/simulation/collection-task0
alohamini-mujoco replay ~/Alohamini_workspace/simulation/collection-task0
```

`TRAJECTORY_JSON` 遵循下方的具名轨迹格式；当前无自动专家策略。`collect` 在初态稳定后以 20 Hz 采样轨迹位置，每条命令保持 50 个 1 ms 物理步，目标速度为零。它与 `follow` 的逐物理步连续参考插值是不同执行协议。

每帧保存控制前的关节位置、速度、RGB 和相机位姿；`actions` 是该周期的绝对 rad/m 位置命令，`states` 保存 N+1 个完整积分状态。默认三路 128×128 RGB 无损压缩；`--cameras` 和 `--image-size` 可显式配置。相机使用 OpenGL 坐标，图像原点在左上角，无额外翻转；记录内参与逐帧外参。

正式收录要求原始 `On` 谓词成立、双指接触至少 0.1 s、接触期间抬升至少 2 cm、释放后稳定至少 0.5 s，并通过关节速度、限位、穿透及完整状态回放检查。原始任务成功与质量检查分别记录。固定体重叠或初态已成功时拒绝启动采集。

通过的轨迹保存在 `episodes/`，失败轨迹在 `rejected/`，失败退出码为 1。目录包含可搬移模型、来源清单、轨迹、HDF5 与 `dataset.json`；禁止覆盖已有目录。回放检查文件校验值与 MuJoCo 版本，使用保存的命令重新积分，不播放录制位置。

初态 0–39 属于训练组，40–49 属于评估组；同一任务和初态的所有扩增须保持同组。这是机器人适配协议。当前为原始仿真格式，尚不能直接与 Host 归一化动作数据混合；须通过具名关节及真机标定显式转换。

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

### 原始 STEP

```bash
python -m pip install -e './simulation/mujoco[cad]'
alohamini-mujoco cad-inspect "${STEP_FILE:?请设置 STEP 路径}" \
  --output ~/Alohamini_workspace/simulation/cad/assembly
alohamini-mujoco cad-tessellate \
  --cad-bundle ~/Alohamini_workspace/simulation/cad/assembly \
  --definition "${CAD_DEFINITION:?请选择 cad.json 中的零件定义 ID}" \
  --linear-deflection-m 0.00005 --angular-deflection-rad 0.1 \
  --output ~/Alohamini_workspace/simulation/cad/part-mesh
```

`cad-inspect` 保留源 STEP、逐零件 BRep、装配实例变换、有效性和实体数量。`cad.json` 的长度、变换平移和体积使用 m/m³；BRep 文件明确保留 OCCT 的 mm 单位。

`cad-tessellate` 仅接受有效实体，每个原始实体及其空腔整体离散，输出零件定义坐标系中的 float64 米制 NPZ 网格。检查闭合性和朝向，记录离散参数、缝合位移与校验值；不自动将开口表面实体化。弦高参数不是经独立验证的表面误差上界。CAD 输出尚未映射到机器人连杆，不能直接作为 `--collision-bundle`。

### 零件坐标校验

```bash
alohamini-mujoco cad-register \
  --mesh-bundle ~/Alohamini_workspace/simulation/cad/part-mesh \
  --link "${URDF_LINK:?请选择 URDF 连杆}" \
  --component-index "${VISUAL_COMPONENT_INDEX:?请选择视觉网格的连通分量序号}" \
  --T-link-part "${T_LINK_PART:?请提供零件到连杆的 4x4 JSON 刚体变换}" \
  --tolerance-m 0.00025 --max-samples 1000000 \
  --output ~/Alohamini_workspace/simulation/cad/part-registration
```

`T_link_part` 将零件定义坐标变换到 URDF 连杆局部坐标，平移单位为 m；程序验证给定变换，不自动拟合。`--visual-index` 默认 0；分量序号对应 `collision.components(meshes.binary_stl(原始STL字节))`，在缝合之前确定，并与源网格校验值绑定。

检查覆盖两个网格的全部三角面，使用自适应细分给出双向最大表面距离的 float64 上下界。`--max-samples` 是每个方向的查询预算，预算不足记为 `inconclusive`，返回失败。输出 `registration.json`、连杆坐标下的 CAD 网格和所选视觉分量。校验仅适用于所选零件与视觉分量，不包含解析 CAD 离散误差、其他零件或实物尺寸，也不替换机器人碰撞体。

### STL 候选

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

分解采用 CoACD，不限制凸体数量，不做减面或膨胀。MSH 使用 float32，在保存后的坐标上重建凸包边界三角形，记录坐标量化误差。`--tolerance-m` 是算法的凹度目标，不代表表面误差上界。精度缝合位移最多 50 nm；仅将完整、朝外且无潜在嵌套的闭合表面组拆开，模糊空腔保留原分组。非流形 CAD 默认拒绝，`--nonmanifold remesh` 显式生成重网格候选，须另行验证几何变化。退化闭壳和无法归属的朝内闭壳直接拒绝，不能用重网格凭空赋予厚度。当前固定夹爪 CAD 默认预检会拒绝。

`collision-merge --bundles 目录1 目录2 --output 新目录` 合并替换不同连杆的候选，保留全部凸体及各自生成参数；重复连杆拒绝合并。

`collision-export --collision-bundle 候选目录 --output 新目录` 导出可搬移的 URDF/SRDF/OBJ，每个凸体对应独立碰撞元素，保留连杆坐标、关节和惯量。`collision-check-sapien --asset 导出目录 --python SAPIEN环境的Python路径 --output 新JSON文件` 从 SAPIEN CPU 后端读回实际凸体，检查数量及逐凸体凸包 Hausdorff 距离；超过 1 μm 返回失败。此项只检查几何，不保证两个引擎的动力学一致；加载成功不代表通过验收。

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

`solve_ik_targets(robot, {"left": T_world_left_tcp, "right": T_world_right_tcp}, movable_joints=具名关节列表)` 支持双臂联合求解；显式加入 `vertical_move` 才允许升降轴参与。`reference_data` 复制实际场景状态，保留稳定后的物体位姿用于接触诊断；输入状态不会被修改。求解受 URDF 关节边界约束，局部求解失败不构成不可达证明。

预览相机用于检查资产；机器人镜头内外参、任务物体接触参数和真机执行器参数须分别标定。此接口不执行真机命令。
