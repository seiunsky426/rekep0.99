# ReKpiper

ReKpiper 是把官方 [ReKep](https://github.com/huangwl18/ReKep) 从
OmniGibson 演示环境接到 **Piper 真机 + 两台固定 RealSense D435** 的 ROS1
复现工程。目标流程是：

```text
语言指令 + 双相机 RGB-D
  → 关键点/物体感知
  → 生成 ReKep 约束
  → 子目标与路径优化
  → Piper IK、碰撞与关节安全检查
  → 执行 0.1 s 短轨迹
  → 重新感知并闭环规划
  → 抓取或释放
```

> 当前是 v0.1 源码版。代码提供了完整的真机工作流和安全接口，但仓库中的
> 相机外参、工作区、夹爪基线、模型依赖和性能验收默认均未批准，因此不能把
> 本仓库视为下载后即可自主运行的真机发行版。

PathSolver 逐次代价/控制点自动日志、RViz 轨迹高亮与一键 IK 预览，见
[PathSolver 诊断工具](docs/PATH_SOLVER_INSPECTOR.md)。启动：
`source setup.bash && roslaunch rekpiper_planning path_solver_inspector.launch`。

## M5 进度（2026-09-14）

**M5.1–M5.4 已完成本次手动观测验证；M5.5/M5.6 按用户要求后置。**
RS1 独立负责 SAM/Cutie 掩膜、DINOv2 关键点跟踪和对象身份维护；RS3 仅提供
点云与辅助视角，不参与掩膜或连续跟踪。

| 子项 | 当前状态与证据 |
|---|---|
| M5.1 身份注册 | 已完成：7 个桌面分割区域全部注册，试验中 UUID 均保持不变 |
| M5.2 RS1 掩膜与可见状态 | 已完成：红色圆柱遮挡及恢复时，状态发生 `FREE_TRACKED → LOST → FREE_TRACKED` |
| M5.3 参考特征初始化 | 已完成：按 M4 快照采集时刻选择 RS1 缓存帧，初始化 6 个固定关键点描述子 |
| M5.4 在线三维关键点与轨迹 | 已完成：保留 K5–K10 编号；手动移动红色圆柱时 K5 跟随，遮挡期间断开轨迹，重新可见后恢复 |
| M5.5 夹爪与生命周期基线 | 后置：留到后续真机执行验证夹持、释放和滑落判据 |
| M5.6 附着几何 | 后置：留到后续真机执行验证关键点与物体点云随夹爪传播 |

本轮记录 571 个跟踪输出，实测约 **4.99 Hz**，采集到接收延迟中位数 **0.527 s**、
P95 **0.649 s**。已提供 RS1 彩色轨迹窗口、保留真实 RGB 的双视角点云 RViz 显示，
并保存三维轨迹 CSV、对象状态 JSONL 和试验视频。同期 252 项 Catkin 测试及仓库、
launch 路径审计通过；离线检查与现场观测证据分别记录。

本次验证范围为单个物体手动移动、遮挡和身份恢复；遮挡边缘仍有短暂三维跳点，
尚未达到 20 Hz 目标，也未完成多物体同时快速移动验证。当前输出
`motion_allowed=false`，本轮结果不代表夹爪、附着或绝对标定精度验收通过。
详细结果及轨迹见 [M5 进度与验收记录](docs/M5_PROGRESS.md)，阶段标注见
[部署流程基线](docs/REKEP_DEPLOYMENT_WORKFLOW_BASELINE.md)。

## M6 进度（2026-09-14）

**M6 已按“获取新地图 → 规划短段 → 后续执行 → 停下更新”的分步实验范围跑通。**
每次按需清图重采，在建图节点内部生成完整 58×68×48 距离栅格（189,312 点），
保留未知区域标记和独立地图 UUID。连续三次耗时 **2.67 / 2.20 / 2.86 秒**，全部成功。
M5 全程未暂停，结束时 6 个关键点有效、7 个对象 UUID 保持；RS3 继续只参与几何观测。

距离数组已接入规划读取接口及官方 ReKep 距离查询；向上 3 cm 的离线预览生成了 7 个 IK 路点。
运动中的掩膜、整臂碰撞与实际跟随、夹爪附着和全面精度验证后置；本轮未发送运动指令，
`planning_safe=false`。264 项 Catkin 测试及仓库、launch 路径审计通过。
使用指令和证据见 [M6 分步规划距离场](docs/M6_PROGRESS.md)。

## M7 进度（2026-09-14）

**约束生成 API 已切换为阿里云 DashScope 的 `qwen3-vl-plus`；在线生成验证仍暂停。**
配置位于 `src/rekpiper_bringup/config/system.yaml`，凭据从 `DASHSCOPE_API_KEY`
环境变量读取。保留官方 ReKep 提示词、AST/数值校验、程序哈希与人工批准流程；
新响应使用 `vlm_raw_response.txt`，批准工具仍兼容原有 GPT-4o 存档。

Qwen 请求格式、凭据选择和错误处理的离线检查通过，Catkin 全套 **266 项测试通过**。
真实生成请求被自动审批拦截：需要用户明确授权将保存的 RS1 编号图和验证提示发送到
DashScope。尚未取得 Qwen 返回，不能认定生成质量合格，M7.1–M7.5 的在线验收保持未完成。
本次证据保存在 `runtime/evidence/m7_qwen_20260914T060049Z/`，未生成或批准运动任务。

## ReKpiper 与官方 ReKep 的区别

| 项目 | 官方 ReKep 公开代码 | ReKpiper |
|---|---|---|
| 运行对象 | OmniGibson 中的 Fetch | Piper 六轴机械臂与原厂夹爪 |
| 环境状态 | 直接读取仿真真值 | 双 D435 RGB-D、DINOv2、SAM/Cutie 与多视角融合 |
| 规划核心 | `SubgoalSolver`、`PathSolver` | 保留固定版本的官方核心，并增加真机代价与 Piper URDF IK |
| 碰撞信息 | 仿真 SDF | 实时深度融合、机器人分割和工作区 SDF |
| 执行方式 | 仿真环境 `step()` | ROS1 短时域关节轨迹、Piper CAN 与夹爪 action |
| 闭环方式 | 每个仿真阶段重新读取状态和求解 | 每段短轨迹后重新获取真实状态、复核约束并重规划 |
| 真机安全 | 不属于公开演示重点 | 稠密路径复核、全路径 IK、关节跳变、全臂碰撞和失败关闭验收 |

### 修改方案与代码位置

ReKpiper 不直接改写官方 `subgoal_solver.py` 和 `path_solver.py`；原始代码保存在
`third_party/ReKep` 并做哈希校验，真机修改通过 ROS 节点和适配层完成：

| 环节 | ReKpiper 相对官方 ReKep 的修改 | 主要代码位置 |
|---|---|---|
| 总体入口与闭环 | 将官方 `main.py` 的仿真阶段循环改为 ROS action；每次只执行短轨迹前缀，然后重新读取关键点、SDF、关节和抓取状态，支持阶段推进与约束失败回退 | [`closed_loop_node.py`](src/rekpiper_execution/scripts/closed_loop_node.py)、[`coordinator.py`](src/rekpiper_execution/src/rekpiper_execution/coordinator.py)、[`trajectory.py`](src/rekpiper_execution/src/rekpiper_execution/trajectory.py) |
| 约束生成 | 保留官方 Python 约束格式和 prompt，默认调用 Qwen VL，增加 AST 白名单、程序哈希和人工批准，禁止模型生成的代码访问文件或网络 | [`vlm_program_generator.py`](src/rekpiper_planning/src/rekpiper_planning/vlm_program_generator.py)、[`official_program.py`](src/rekpiper_planning/src/rekpiper_planning/official_program.py)、[`program_server_node.py`](src/rekpiper_planning/scripts/program_server_node.py) |
| 双相机与标定 | 用两台固定 D435 的彩色图和深度生成点云，通过 eye-to-hand 外参变换到 `base_link`，裁剪工作区后融合；增加 ArUco 18 标定和独立复核 | [`rgbd_projection_node.py`](src/rekpiper_camera/scripts/rgbd_projection_node.py)、[`dual_workspace_cloud_fusion_node.py`](src/rekpiper_camera/scripts/dual_workspace_cloud_fusion_node.py)、[`solve_dual_aruco18_dataset.py`](src/rekpiper_calibration/scripts/solve_dual_aruco18_dataset.py) |
| 关键点与物体状态 | 官方公开代码直接读取仿真状态；这里由 RS1 的 SAM 生成物体掩码、Cutie 跨帧跟踪掩码、DINOv2 跟踪三维关键点，并生成带版本绑定的场景快照；RS3 仅提供点云和辅助视角 | [`task_perception_node.py`](src/rekpiper_perception/scripts/task_perception_node.py)、[`dual_dino_tracker_node.py`](src/rekpiper_perception/scripts/dual_dino_tracker_node.py)、[`cutie_backend.py`](src/rekpiper_perception/src/rekpiper_perception/cutie_backend.py)、[`scene_snapshot_node.py`](src/rekpiper_perception/scripts/scene_snapshot_node.py) |
| 子目标与路径优化 | `official_exact` 直接加载固定版本的官方求解器并保留逐阶段 warm-start；`paper_real` 是单独的可选实现，在官方几何工具上加入上一轮一致性、桌面净空和全臂代价 | [`upstream.py`](src/rekpiper_planning/src/rekpiper_planning/upstream.py)、[`realtime_planner.py`](src/rekpiper_planning/src/rekpiper_planning/realtime_planner.py)、[`paper_real_solver.py`](src/rekpiper_planning/src/rekpiper_planning/paper_real_solver.py) |
| Piper IK 与碰撞 | 将官方面向仿真机器人的 IK/SDF 接口替换为 Piper URDF 数值 IK；对稠密笛卡尔路径逐点求 IK，并使用 Piper 全臂采样点检查实时 SDF | [`piper_urdf_ik.py`](src/rekpiper_planning/src/rekpiper_planning/piper_urdf_ik.py)、[`piper_collision_sampling.py`](src/rekpiper_planning/src/rekpiper_planning/piper_collision_sampling.py)、[`trajectory_audit.py`](src/rekpiper_planning/src/rekpiper_planning/trajectory_audit.py)、[`sdf_conventions.py`](src/rekpiper_mapping/src/rekpiper_mapping/sdf_conventions.py) |
| 地图与机器人剔除 | 使用双 D435 深度构建 nvblox/ESDF，并用 cuRobo 机器人分割及抓取物附着几何避免把机械臂自身写入障碍地图 | [`nvblox_mapping_node.py`](src/rekpiper_mapping/scripts/nvblox_mapping_node.py)、[`curobo_robot_segmenter_node.py`](src/rekpiper_mapping/scripts/curobo_robot_segmenter_node.py)、[`sdf_grid_snapshot_node.py`](src/rekpiper_mapping/scripts/sdf_grid_snapshot_node.py) |
| 真机执行与夹爪 | 将官方 `environment.execute_action/open_gripper/close_gripper` 替换为 Piper 关节轨迹桥、CAN 控制和夹爪 action；AnyGrasp 只在抓取阶段用于候选选择 | [`piper_trajectory_bridge_node.py`](src/rekpiper_execution/scripts/piper_trajectory_bridge_node.py)、[`piper_gripper_action_node.py`](src/rekpiper_execution/scripts/piper_gripper_action_node.py)、[`keypoint_anygrasp_selector_node.py`](src/rekpiper_grasp/scripts/keypoint_anygrasp_selector_node.py) |
| 启动与失败关闭 | 新增 shadow/autonomous 模式、预检、硬件监督、签名验收和运行性能门槛；标定、地图或反馈过期时不下发运动 | [`system.launch`](src/rekpiper_bringup/launch/system.launch)、[`start_rekpiper.py`](src/rekpiper_bringup/scripts/start_rekpiper.py)、[`hardware_supervisor_node.py`](src/rekpiper_bringup/scripts/hardware_supervisor_node.py) |

官方快照位于 `third_party/ReKep`，固定到提交
[`63c43fd`](https://github.com/huangwl18/ReKep/commit/63c43fdba60354980258beaeb8a7d48e088e1e3e)，
并由哈希清单校验；ReKpiper 的真机适配代码主要位于 `src/rekpiper_*`，不直接
修改这份官方快照。

## 仓库内容

- `src/`：13 个 ROS1 包，包含双相机、标定、感知、规划、执行和安全验收。
- `third_party/`：官方 ReKep 固定快照及第三方版本/模型清单。
- `docs/`：真机部署与验收说明。
- `tools/`：仓库、launch 路径和运行环境检查工具。
- `requirements.*`、`setup.bash`：Python 锁定依赖与环境入口。

第一版 GitHub 仓库不应只放 `src/` 和 README；应提交上述目录与根目录配置，
但不要提交 `build/`、`devel/`、`install/`、模型权重、标定数据、密钥或本机 IDE
配置。

## 仍需完成的工作

当前代码框架已经覆盖完整流程，但真机自主执行仍被安全门槛主动禁止。需要按以下
顺序完成：

- [x] **完成本机 M0 软件基线**：正式 runtime、三份依赖锁、六个模型/扩展、
  上游源码恢复和 rs1/rs3 生产配置已验收；13 包构建成功，209 项测试全部通过。
  复查命令和证据见 [M0 验收记录](docs/M0_2_TO_M0_5_COMPLETION.md)。
- [ ] **完成现场运行资产批准**：使用已记录的版本、哈希和许可证据，按部署合同
  形成现场批准材料；`third_party/VENDOR.lock.yaml` 仍保留未批准状态。
- [ ] **完成双 D435 标定**：采集足够的 ArUco 18 多姿态数据，求解并交叉复核
  `base_link <- camera` 外参。当前 rs1/rs3 文件仍是 `UNCALIBRATED`，不能用于
  自主运动。
- [ ] **验收工作区和桌面**：实测 Piper 可达范围、桌面高度和禁入区域，验证两台
  相机的共同覆盖范围，并把工作区配置从仅测量状态批准为可精密操作。
- [ ] **完成 Piper 与夹爪基线验收**：验证 CAN、关节反馈、零位、急停、限位和
  Teaching mode 切换；采集空夹、夹持和释放数据，确定夹爪开度/力矩判据。
- [ ] **完成复杂场景感知验收**：M5 已通过本次单物体手动移动、遮挡和恢复验证；
  仍需验证 RS1 的 SAM/Cutie 掩膜及 DINOv2 关键点跟踪在多物体快速移动、反光和
  机械臂运动时的稳定性，并改善遮挡边缘跳点与处理延迟。
- [ ] **验收实时地图和碰撞模型**：确认 nvblox/ESDF 的尺度和符号约定正确，机械臂
  与抓取物被正确剔除/附着，并通过全路径 Piper IK、关节跳变及全臂碰撞测试。
- [ ] **标定 `paper_real` 求解权重**：当前候选配置仍为 `approved: false`；需要在
  代表性场景中校准一致性、桌面净空等权重并生成批准记录，也可先用
  `official_exact` 做 shadow 对照测试。
- [ ] **完成性能与闭环验收**：测量感知、SDF、规划和轨迹下发延迟，确认 10 Hz
  规划及 0.1 s 短轨迹前缀不会断流。当前软件和硬件性能文件均为
  `UNAPPROVED`。
- [ ] **分阶段真机复现**：依次完成离线回放、shadow、单步运动、抓取/释放和完整
  指令任务；全部证据签名通过后，才允许启用 `autonomous` 与硬件命令。
- [ ] **确定公开发布许可**：为自有代码选择许可证，更新 `rekpiper_*` 包中的
  `Proprietary` 声明，并核对官方 ReKep、Piper SDK 和第三方模型的再分发条件。

具体命令和验收产物见 [docs/deployment.md](docs/deployment.md)。

## 基本构建与检查

目标环境为 Ubuntu 20.04、ROS Noetic、Python 3.8：

```bash
git clone https://github.com/1zjj/Rekep_v1.0.git
cd Rekep_v1.0

python3 -m venv runtime
source runtime/bin/activate
pip install --require-hashes -r requirements.lock
pip install --require-hashes -r requirements.platform.lock
pip install --no-build-isolation --require-hashes -r requirements.models.lock

source setup.bash
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DPYTHON_EXECUTABLE="$REKPIPER_RUNTIME_ROOT/bin/python" \
  -DNOSETESTS="$REKPIPER_RUNTIME_ROOT/bin/nosetests"
source setup.bash

python3 tools/check_repository.py --source-root "$PWD"
python3 tools/check_launch_paths.py --package-root "$PWD/src" --dump-shadow
catkin_make run_tests
catkin_test_results build/test_results
```

当前机器的独立环境、CUDA/ROS 辅助依赖和离线复查命令见
[M0.1 环境说明](docs/M0_1_ENVIRONMENT.md)及 [M0 完整验收记录](docs/M0_2_TO_M0_5_COMPLETION.md)。

DINOv2、SAM、Cutie、cuRobo、nvblox 和 AnyGrasp 的代码/权重不随仓库分发，
需要按照 `third_party/VENDOR.lock.yaml` 在外部 runtime 中准备；其中 AnyGrasp
还需遵守其授权要求。更完整的真机验收流程见 [docs/deployment.md](docs/deployment.md)。

## 安全说明与致谢

不要绕过 `UNCALIBRATED`、`approved: false` 或签名验收检查直接启动自主控制。
首次真机运行必须有人值守、急停可用，并先完成双相机外参、Piper 工作区、夹爪
基线和碰撞地图验证。

本项目基于 Huang 等人的 ReKep 工作。算法、论文和官方演示代码请引用并遵循
[ReKep 官方仓库](https://github.com/huangwl18/ReKep)的说明与使用条款。

本仓库当前没有统一的开源许可证，`rekpiper_*` 包仍标记为 `Proprietary`。
公开发布前应先确定自有代码的许可证，并核对官方 ReKep、Piper SDK 及模型依赖的
再分发条款；公开 GitHub 仓库不等于自动授予开源许可。
