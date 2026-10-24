# MuJoCo

支持 AlohaMini2Pro 固定底盘的交互查看、模型检查、完整位姿 IK 和动力学关节轨迹执行。底盘三个虚拟关节及三个轮轴固定；双臂、夹爪和升降轴保留重力、关节限位与电机力限制。

## 安装

在仓库根目录执行：

```bash
conda activate alohamini
python -m pip install -e ./simulation/mujoco
```

运行时固定 MuJoCo 3.12.0；LIBERO 原始导出环境独立固定 3.8.1。已有数据须使用记录的 MuJoCo 版本回放，不做静默跨版本迁移。

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

支持 10 条 Spatial 任务，各保留 50 个具名物体初态；不迁移 Panda 关节或 7 维动作。快照包含原物体网格、纹理、分块碰撞体、惯量、接触参数、BDDL、导出环境版本和文件校验值；导出版本与运行版本分别记录。`--task-index` 为 0–9，`--reset-index` 为 0–49。其他 LIBERO 套件和谓词尚不支持。

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

`cad-tessellate` 仅接受有效实体，每个原始实体及其空腔整体离散，输出零件定义坐标系中的 float64 米制 NPZ 网格。相邻曲面通过原始 CAD 共享边的参数和顶点身份拼接，不按距离合并独立边；边界点取自原始顶点或三维边曲线，逐点验证局部原生容差。记录 `tessellation_method`、`boundary_binding`、闭合性、朝向及校验值。退化三角形仅在其塌缩结果已被保留边界覆盖时移除，不补面或封洞。弦高参数不是经独立验证的表面误差上界。CAD 输出尚未映射到机器人连杆，不能直接作为 `--collision-bundle`。

`occt-topological-boundaries-v2` 可细分同一 CAD 面内部的粗边，使其与已有节点一致；须保留原 CAD 边界，表面变化不得超过该面的原生容差及浮点误差。修复不新增顶点或封口，记录于 `face_interior_conformity`。非流形顶点仅可分离连接索引，保持每个有向三角形不变，记录于 `vertex_fan_separation`；重复面、真实孔洞或无法证明的连接继续拒绝。

整机逐件检查：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 alohamini-mujoco cad-tessellate-all \
  --cad-bundle ~/Alohamini_workspace/simulation/cad/assembly \
  --output ~/Alohamini_workspace/simulation/cad/library
```

每个零件在独立进程中离散，`library.json` 逐件原子更新，保留全部装配实例变换及 `meshed`、`surface_only`、`invalid_brep`、`meshing_failed`、`worker_failed` 状态。`complete` 表示全部定义已检查，`fully_meshed` 才表示全部成功；存在未通过项时退出码为 1，不能将缺失零件当作可用碰撞模型。

`--reuse-library 旧目录` 可向新目录复用已有网格，严格检查同一 STEP、BRep 校验值、定义坐标、OCCT 版本、离散方法版本、参数及全部网格文件；不兼容缓存返回失败，不覆盖旧输出。派生装配中未变的零件可复用，网格原始清单保留，当前装配绑定记录于新 `library.json`。中断后的目录也可作为只读缓存；此操作只生成来源网格，不决定刚体归属或接触近似。

### STEP 装配查看

```bash
alohamini-mujoco cad-scene \
  --cad-bundle ~/Alohamini_workspace/simulation/cad/assembly \
  --T-world-cad "${T_WORLD_CAD:?请提供 CAD 到查看坐标系的 4x4 JSON 刚体变换}" \
  --output ~/Alohamini_workspace/simulation/cad/assembly-view
python -m mujoco.viewer \
  --mjcf ~/Alohamini_workspace/simulation/cad/assembly-view/model.xml
```

默认包含全部叶零件；`--occurrence 装配或零件的完整实例ID` 可选择子装配。保留原实例变换、全部原始面和非闭合外壳，每种零件独立进程离散；缺面则拒绝导出。`scene.json` 记录面索引、来源校验值、网格警告和 MuJoCo 坐标读回结果。此静态参考关闭接触，无运动关节，不是碰撞候选或动力学模型。

### 方管解析碰撞组件

```bash
alohamini-mujoco cad-prism \
  --cad-bundle ~/Alohamini_workspace/simulation/cad/assembly \
  --definition "${CAD_DEFINITION:?请选择空心方管零件的定义ID}" --axis z \
  --output ~/Alohamini_workspace/simulation/cad/beam-component
python -m mujoco.viewer \
  --mjcf ~/Alohamini_workspace/simulation/cad/beam-component/component.xml
```

`--axis` 是零件定义坐标系中的贯通方向。命令只接受单个、轴向平面组成的空心矩形柱体，生成保留贯通空腔的 4 个原生 box。原实体与组合体的两个方向布尔差集均须为空，检查精度受 OCCT 及源 CAD 容差限制；MuJoCo 另行验证空腔与四面管壁。额外孔、端盖、曲面或差集残留均拒绝，不丢弃几何特征。

输出 `component.json` 和可直接查看的静态 `component.xml`。组件位于 CAD 零件定义坐标系，保留装配实例变换；须明确刚体归属和零件到连杆的变换后，才能组合进整机模型。组件不是完整连杆替换清单，不能直接传给 `--collision-bundle`。

### 零件坐标校验

缺少实体 CAD 时，可从现有二进制 STL 提取闭合参考网格；需要 `[geometry]` 依赖：

```bash
alohamini-mujoco mesh-source \
  --link "${URDF_LINK:?请选择 URDF 连杆}" \
  --component-index "${VISUAL_COMPONENT_INDEX:?请选择视觉网格分量}" \
  --output ~/Alohamini_workspace/simulation/cad/mesh-reference
```

输出 `mesh.json` 与闭合网格，坐标为应用 URDF scale 后的视觉局部坐标；`T_link_part_hint` 给出其到连杆的变换。默认接缝位移须落在每个原坐标的 float32 舍入区间内；非流形连接仅允许分离顶点索引，逐面保持原三角形坐标和朝向。重复面、无法闭合的边界及需要删面、补面的情况均拒绝。`source_format: urdf-binary-stl` 不包含 BRep，原始 CAD 离散误差未知。后续使用相同的配准、碰撞生成和探针命令。

完整反向开边可显式使用 `--seam-budget-m 0.000001`，允许最大 1 μm 的几何修订；默认精度规则保持不变。仅配对互为唯一对应的整条边，不补面、删面或新增拓扑顶点，原三角形朝向保持。输出标记 `source_format: urdf-binary-stl-sewn`，记录边配对、实际位移及逐面对应的双向连续表面误差上界。超预算、缺面、重复面、塌缩或歧义继续拒绝；该预算不代表原始 CAD 或实物精度，修订后仍需检查功能间隙与接触。

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

多个闭壳在原 STL 中共享边时，可额外指定 `--surface-index`，按仅有两面相邻的边划分所选连通分量。选择仅引用原三角面，不补面或改变坐标；输出保留分区索引校验值，不据此推断独立实体或刚体归属。

检查覆盖两个网格的全部三角面，使用自适应细分给出双向最大表面距离的 float64 上下界。`--max-samples` 是每个方向的查询预算，预算不足记为 `inconclusive`，返回失败。输出 `registration.json`、连杆坐标下的 CAD 网格和所选视觉分量。校验仅适用于所选零件与视觉分量，不包含解析 CAD 离散误差、其他零件或实物尺寸，也不替换机器人碰撞体。

可加 `--measure-resolution-m 0.00005`，独立测量最大距离的上下界，目标区间宽度为 0.05 mm，并记录距离见证点及源、目标三角面序号。测量不改变验收预算或配准结果；查询预算不足时测量状态为 `inconclusive`。

安装平面可单独核对：

```bash
alohamini-mujoco cad-place \
  --cad-bundle ~/Alohamini_workspace/simulation/cad/assembly \
  --binding "${DATUM_BINDING:?请提供平面对应配置的 JSON 路径}" \
  --output ~/Alohamini_workspace/simulation/cad/part-placement
```

配置包含 `schema_version: 1`、`cad_manifest_sha256`、`definition`、`link`、`visual_index`、`visual_sha256`、`R_link_part` 和 `datums`。每项 datum 包含唯一 `name`、`role`（`fit` 或 `check`）、`cad_face` 和 `visual_triangles`。CAD 面序号采用 `TopExp_Explorer(shape, TopAbs_FACE)` 的从零索引；视觉三角形采用原始 STL 文件中的从零索引，各项不得重用三角形或 CAD 面。

旋转须显式给定；至少 3 个定位平面约束全部平移轴，并提供独立检查平面。程序仅拟合平移，检查平面不参与拟合。误差预算来自各 CAD 面及其边界的原生容差、STL float32 量化和 float64 运算，不包含未知的早期导出误差或实物标定。`placement.json` 分别记录定位、独立检查及总体结果；任一未通过则退出码为 1。此检查不替代整面形状校验，也不决定刚体归属。

### 闭合 CAD 碰撞体与接触检查

```bash
python -m pip install -e './simulation/mujoco[geometry]'
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 alohamini-mujoco cad-collision \
  --mesh-bundle ~/Alohamini_workspace/simulation/cad/part-mesh \
  --registrations "${PART_REGISTRATION:?请指定完整连杆的配准目录}" \
  --concavity-m 0.0005 --tolerance-m 0.0005 --refine --merge-verified \
  --output ~/Alohamini_workspace/simulation/collision/cad-part
```

默认仅接受单个闭合 CAD 实体及覆盖整条连杆全部视觉三角形的已通过配准；不推断刚体归属、镜像或缩放。多个连杆须分别提供配准目录。CoACD 关闭重网格、减面和膨胀；`--concavity-m` 是分解目标，`--tolerance-m` 是独立的实体距离验收预算。

多零件连杆可加 `--component-only`，针对所选完整视觉分量导出 `component.json`。此模式不接受 `--surface-index` 分区，也不生成 `collision.json`；零件包不能直接用于整机替换、跟随或数采。加载时同时校验分量选择、当前源资产与碰撞体文件，不把零件验收解释为整条连杆验收。

直接检查一个或多个已验收零件包：

```bash
alohamini-mujoco component-view \
  --component-bundles "${COMPONENT_BUNDLE:?请选择包含 component.json 的目录}" \
  --initial-positions '{"vertical_move":0.2}' --links left_Fixed_Jaw
```

`--component-bundles` 可接多个目录。右侧关节滑块只改变运动学姿态；F8 切换所选连杆的原视觉网格与新增碰撞组件，F7 叠加，F9 复位，F6 显示坐标系。同一视觉分量不能重复加入。原碰撞体保留但不显示；窗口不运行动力学，也不代表完整连杆覆盖、抓取或整机验收。

验收检查 CAD 与 float32 碰撞凸体并集的双向实体包容关系，使用内接于误差球的凸多面体作 Minkowski 膨胀核，并记录布尔核容差。此项验证材料覆盖，不能替代接触法向、功能间隙或实物标定。`--refine` 仅裁剪超过包络的凸体，在六个误差区域边界平面中选择保留体积最大的合格裁剪，再检查整个结果。验收失败只保存 `qualification.json`，不生成可加载的 `collision.json`。

输出的 `decomposition/` 可通过 `--reuse-decomposition 目录` 复用；缓存须匹配源网格、定义坐标、CoACD 版本和全部参数。加载与合并碰撞候选时检查验收记录与凸体文件校验值的绑定。

`--merge-verified` 在相同误差包络内尝试合并相邻凸体，最后重新检查完整几何与网格有效性。数值退化或覆盖失败时回退合并。每次构建最多检查 20,000 对，结果记录保留与回退数量、搜索是否结束；不保证凸体数量全局最少。

```bash
alohamini-mujoco collision-probe \
  --mesh-bundle ~/Alohamini_workspace/simulation/cad/part-mesh \
  --collision-bundle ~/Alohamini_workspace/simulation/collision/cad-part \
  --configuration "${CONTACT_FIXTURE:?请指定接触探针 JSON}" \
  --output ~/Alohamini_workspace/reports/simulation/contact-probe.json
```

配置字段为 `schema_version: 1`、`source_mesh_manifest_sha256`、`sphere_radius_m`、`tolerance_m`、`clearances_m` 和 `cases`。每个 case 指定 `link`、`point_link_m`、`outward_normal_link`、`joint_positions`（具名 URDF 关节位置字典列表，m/rad）。探针中心须位于 CAD 材料外；检查原生碰撞距离、接触是否产生，并报告法向偏差。结果仅覆盖指定采样点、关节配置和球体尺寸，不代表完整抓取验收。

检查零件包时，将 `--collision-bundle` 换为 `--component-bundle`。夹具保留所有原机器人碰撞体，只查询新增零件与探针之间的距离、接触；输出标记 `coverage: visual-component`。该夹具禁止动力学跟随和数采，不能用于验证整机抓取。

### 刚体归属与随动检查

```bash
alohamini-mujoco assembly-view \
  --assembly examples/simulation/gripper_assembly.json \
  --initial-positions '{"vertical_move":0.2}' \
  --links left_Fixed_Jaw left_Moving_Jaw
alohamini-mujoco assembly-inspect \
  --assembly examples/simulation/gripper_assembly.json --samples 17 \
  --output ~/Alohamini_workspace/reports/simulation/rigid-assembly
```

示例为夹爪装配候选，实物花键相位、轴向贴合及内部轴件归属尚未验收。

装配定义使用 schema 1、`robot_model`、`units: m`、`source_description_sha256` 和 `sources`。每个来源指定 `link`、`visual_index`、完整 `source_visual` 校验记录及 `components`；所选视觉网格的每个连通分量须恰好出现一次。分量指定 `component_index`、`owner` 和 `reference_positions`，后者须显式列出原连杆至目标连杆路径上的全部活动关节，使用 rad/m；归属不变时为 `{}`。

程序由参考姿态的 URDF 正运动学计算零件至目标刚体的变换，保持参考姿态下的世界坐标、全部原三角形、原网格缩放与视觉原点。来源变化、分量遗漏、参考姿态越界或与锁定配置冲突时拒绝加载；安装相位的机械依据由装配定义提供，程序不自动推断。

两条命令均可添加 `--component-bundles 零件目录1 零件目录2`，将已校验的零件碰撞体按相同刚体归属变换。零件包仍只覆盖各自分量，原碰撞体保留，不成为完整连杆替换。检查逐顶点验证原生几何、参考姿态守恒及关节限位采样；`passed` 不代表连续无碰撞或安装、动力学验收。

分量可选 `collision_envelope`，须包含非空 `primitives` 与 `limitations`。每个形状指定 `type`（box/cylinder/sphere）、`parameters` 和 `T_link_geometry`；变换属于原来源连杆坐标，单位 m，不应用视觉网格缩放。box 的 `size` 为三个全长，cylinder 的 `radius`、`length` 为半径与全长，轴为形状局部 Z。程序使用同一参考姿态将其绑定至目标刚体，检查原生类型、尺寸和随动坐标；同一分量不能同时加载凸体包与包络候选。包络仅供检查，不自动获得材料或接触等价认证。

夹爪示例为每个驱动盘配置两个圆柱候选；被动盘保留原几何，需另行添加其有空腔的零件包。安装孔、轴腔等功能间隙须在整机装配中验证，不能以外轮廓误差替代。

窗口使用关节滑条改变姿态，F9 复位；橙/紫等颜色标识转移的零件，绿色表示接收零件的连杆。F8 切换碰撞体，F7 叠加；有零件包或包络时只显示这些候选，否则显示原归属的旧碰撞体。两种模式均不运行物理积分。原 URDF 质量、惯量和自碰撞过滤保留；碰撞覆盖与质量分配未完成前禁止跟随和数采。

接触探针和 LIBERO 静态场景验收检查原生距离与最近点线段的一致性，并用凸体顶点或解析形状的投影区间查找矛盾。异常不修正为通过值：探针拒绝验收，场景在 `invalid_distance_queries` 中记录并阻止数采。这些检查不能发现所有求解器误差，也不能替代独立间隙验证。

### 整机碰撞组织

```bash
alohamini-mujoco collision-organize \
  --cad-bundle ~/Alohamini_workspace/simulation/cad/assembly \
  --output ~/Alohamini_workspace/simulation/collision/organized
alohamini-mujoco view --collisions --paused \
  --collision-bundle ~/Alohamini_workspace/simulation/collision/organized \
  --initial-positions '{"vertical_move":0.2}'
```

`collision.json` 的 `assembly_inventory` 覆盖整棵连杆树，记录各连杆的视觉、碰撞形状、坐标、来源校验值和处理状态。程序从已有碰撞 STL 的连续面序列中恢复闭合凸体；每个源三角形恰好归属一次，不缝合、补面或重新分解。原生基本形状及混合夹爪接触组件保持原定义，惯量继续取自 URDF，完整 SRDF 自碰撞过滤保留。

无法恢复的网格明确保留并标记 `requires_authored_geometry`。此命令整理旧碰撞代理，不应用 STEP 尺寸更新；整机 CAD 映射、形状误差及接触验收仍须完成。加载与合并候选时继续验证整机来源，不能遗漏未替换连杆的几何变化。

### 原生形状与组合体

```bash
alohamini-mujoco collision-fit-primitives \
  --links chest_camera front_camera back_camera wheel1 wheel2 wheel3 \
  --output ~/Alohamini_workspace/simulation/collision/primitives
```

仅接受已有单个 box/cylinder 的连杆，使用完整视觉网格和原碰撞坐标轴计算包络，不使用 PCA 猜测轴向。箱体取固定轴向最小包围盒；圆柱保留轮轴位置和方向，计算所需半径、轴向中心和长度。输出记录全部视觉三角形的包含依据及浮点向外舍入量；不包含物理安全余量，也不代表 STEP 尺寸或接触精度已验收。

碰撞清单 schema 2 支持同一连杆内混合凸网格、box、sphere、cylinder。原生形状须显式提供 `type`、`parameters`、`T_link_geometry`；box 的 `size` 和 cylinder 的 `length` 均为完整米制尺寸，圆柱轴为局部 Z。凸网格顶点已经处于连杆局部坐标，不接受额外缩放或位姿。旧 schema 1 网格候选仍可加载及合并。MuJoCo 和 URDF 导出保留原生形状，质量、惯量及 SRDF 过滤取自机器人原定义。

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

`collision-export --collision-bundle 候选目录 --output 新目录` 导出可搬移的 URDF/SRDF/OBJ，每个形状对应独立碰撞元素，保留连杆坐标、关节和惯量。`collision-check-sapien --asset 导出目录 --python SAPIEN环境的Python路径 --output 新JSON文件` 从 SAPIEN CPU 后端读回实际形状，检查数量、逐凸体凸包 Hausdorff 距离及原生形状距离上界；超过 1 μm 返回失败，原生形状被换成其他类型也返回失败。此项只检查几何，不保证两个引擎的动力学一致；加载成功不代表通过验收。

`collision-export --mesh-unit mm` 将生成的 OBJ 坐标以 mm 编码，通过 URDF 的 `scale="0.001 0.001 0.001"` 恢复物理米制尺寸；机器人位姿和惯性参数仍使用 SI。此选项用于检查后端网格预处理的数值精度，验收阈值保持 1 μm。

`inspect` 的 `convexification` 报告列出每个物理网格的凸包与源网格诊断；非闭合或朝向不一致的网格不提供体积比。夹爪截面报告两指各自接触面的 X 坐标、偏差和命中统计，并比较开口宽度，避免两指同向偏移时误差抵消。`probe_counts` 区分开口配对是否可测量，缺失不等于零误差。`validation.passed` 仅表示模型导入一致性通过，候选仍须验证接触、间隙与动力学性能。碰撞候选不会自动成为默认模型。

## 轨迹执行

模型编译与导出、回放重载使用最高 64 MiB 的临时编译资产缓存；原缓存容量随后恢复。该预算不限制运行模型内存，也不减少几何或物理计算。`follow` 输出墙钟耗时、实际运行倍率和 arena 使用峰值；耗时包含已启用的观测回调。

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
