# ReKpiper 端到端运行链路

## 1. 总目标

“跑通 ReKep”不是让 ROS 节点全部启动，也不是让求解器返回一个 7D 位姿，而是完成下面的真实闭环：

> 将自然语言任务转换为空间约束；从真实相机和机械臂获得当前世界状态；计算满足任务约束且无碰撞的运动；让 Piper 只执行一小段；重新感知、更新地图并重新规划；最终完成抓取、移动和释放，并由反馈证明任务成功。

完整成功需要依次满足以下层级：

1. **依赖可用**：代码、模型、许可证、CUDA 和 ROS 依赖通过预检。
2. **数据可用**：相机、点云、TF、关节、跟踪和地图持续输出真实且新鲜的数据。
3. **程序可用**：自然语言任务被转换成正确、可审查的 ReKep 约束程序。
4. **规划可用**：约束求解、路径求解、IK 和碰撞审计通过。
5. **Shadow 可用**：能够稳定发布有效规划预览，但不发送硬件命令。
6. **硬件可用**：短轨迹能够经过授权发送给 Piper，并获得可信反馈。
7. **任务完成**：抓取、附着、地图重建、释放和所有任务阶段全部完成。

必须始终区分：**接口存在、实时数据已连接、Shadow 规划成功、真机任务验收通过**。

## 2. 主数据流

```text
操作者任务指令
   │
   ├── /rekpiper/perception/task_request
   │          ↓
   │   RGB-D → SAM物体分割 → DINOv2关键点候选
   │                              ↓
   │                        SceneSnapshot
   │
   └── /rekpiper/program/generate
              指令 + 关键点图 + SceneSnapshot
                              ↓
                     GPT-4o生成ReKep约束
                              ↓
                       审查并批准程序

双相机实时数据 ─────────────────────────────┐
Cutie物体掩码 → nvblox TSDF/ESDF → SDFGrid ─┤
双目DINO关键点跟踪 → 当前关键点 ───────────┤
Piper关节反馈 + TF → 当前末端位姿 ─────────┤
                                             ↓
                                   ClosedLoopNode
                                             ↓
                          SubgoalSolver：下一目标位姿
                                             ↓
                          PathSolver：笛卡尔运动路径
                                             ↓
                        Piper IK + 全机械臂碰撞审计
                                             ↓
                      ReKepHorizon：只授权未来0.1秒
                                             ↓
              Shadow：仅预览 / Autonomous：发送Piper
                                             ↓
                             新图像、地图和反馈
                                             └──────↺

抓取阶段：
ReKep抓取关键点 → 最近安全AnyGrasp TCP子目标 → PathSolver → IK/ESDF审计
→ 夹爪闭合 → 接触与附着验证 → 更新物体状态 → 重建地图
```

## 3. 三个入口

### 3.1 进程启动入口

入口文件：

```text
src/rekpiper_bringup/scripts/start_rekpiper.py
```

调用关系：

```text
start_rekpiper.py
  → preflight_cli.py
  → roslaunch rekpiper_bringup system.launch
```

预检负责检查运行目录、模型清单、第三方代码、验收文件和运行模式。`system.launch` 随后启动双相机、感知、跟踪、建图、程序生成、规划和抓取节点。

默认配置为：

```text
mode=shadow
allow_hardware_commands=false
```

因此启动成功只表示 ROS 计算图已经建立，不表示任务已经开始，更不表示机械臂能够运动。

### 3.2 任务感知入口

操作者或上层客户端需要向下面的话题发布任务文本：

```text
/rekpiper/perception/task_request
```

订阅代码：

```text
src/rekpiper_perception/scripts/task_perception_node.py
  TaskPerceptionNode._task_request_callback()
```

该请求触发一次新的 SAM+DINOv2 场景识别和静态快照锁定。任务文字在此主要承担“新任务触发器”的作用；任务语义的主要解释发生在后面的 GPT-4o 程序生成阶段。

### 3.3 执行入口

程序生成并批准后，执行侧需要：

1. 调用 `/rekpiper/execution/arm`；
2. 向 `/rekpiper/execution/execute_rekep` Action 发送 `session_id` 和 `program_sha256`。

入口代码：

```text
src/rekpiper_execution/scripts/closed_loop_node.py
  ClosedLoopNode._execute()
```

`_execute()` 管理任务生命周期。真正持续读取状态、求解和生成短轨迹的是 10 Hz 定时调用的 `ClosedLoopNode._tick()`。

## 4. 模块级完整流通

### 4.1 RGB-D 投影：从像素到机器人三维坐标

**目的**

将相机二维图像和深度转换为 `base_link` 坐标系中的三维点。

**输入**

- RGB 图像；
- 对齐到彩色图的深度图；
- `CameraInfo.K` 相机内参；
- 相机光学坐标系到 `base_link` 的时间同步 TF。

**方法与原理**

```text
像素(u,v) + 深度z + 内参K
        ↓ 深度反投影
相机坐标系三维点(x,y,z)
        ↓ 对应深度时间戳的TF
base_link坐标系三维点
```

代码：

```text
src/rekpiper_camera/scripts/rgbd_projection_node.py
  RgbdProjectionNode._callback()
```

代码不会在 TF 查询失败时退回到旧 TF，从而避免把历史相机位姿错误地用于当前图像。

**输出**

```text
/rekpiper/camera/rs1/points_recognition
/rekpiper/camera/rs3/points_recognition
```

消息类型为 `sensor_msgs/PointCloud2`。

**下一模块**

- 任务感知；
- 双目关键点跟踪；
- nvblox 建图；
- AnyGrasp 当前使用 RS1 工作空间点云。

### 4.2 SAM+DINOv2：从场景到关键点

**目的**

把复杂场景压缩为少量能够描述任务空间关系的三维关键点 `K0...Kn`。

**输入**

- `/rekpiper/perception/task_request`；
- RS1 RGB 图像；
- RS1 组织点云。

**方法与原理**

1. SAM 生成物体实例掩码；
2. 删除无效深度和工作空间外区域；
3. DINOv2 提取视觉特征；
4. 联合视觉特征和 XYZ 位置进行聚类；
5. 生成候选关键点，并绑定物体的 `rigid_group_id`；
6. 在图像上标注 `K0...Kn`，供 GPT-4o 使用。

代码：

```text
src/rekpiper_perception/scripts/task_perception_node.py
  TaskPerceptionNode._callback()
```

**输出**

- `Keypoint3DArray`：关键点三维坐标和刚体组；
- `InstanceGeometryArray`：物体实例和几何范围；
- `candidate_image`：标注关键点编号的图像；
- 场景物体掩码。

**下一模块**

- `scene_snapshot_node.py`；
- `program_server_node.py`；
- 物体注册与在线跟踪。

### 4.3 SceneSnapshot：冻结任务初始场景

**目的**

确保关键点编号、物体分组、候选图像和后续生成的程序属于同一个场景，防止规划期间编号或布局被静默替换。

**输入**

- `Keypoint3DArray`；
- `InstanceGeometryArray`；
- 图像、掩码和时间戳。

**代码**

```text
src/rekpiper_perception/scripts/scene_snapshot_node.py
src/rekpiper_perception/scripts/snapshot_object_registrar_node.py
```

**输出**

```text
/rekpiper/perception/scene_snapshot
```

`SceneSnapshot` 主要包含：

- `snapshot_id`；
- 初始关键点；
- 物体实例；
- 相机和图像绑定；
- `valid`；
- `immutable_layout`。

快照注册器还会为每个刚体组创建对象身份、初始化在线跟踪并请求安全地图重建。

**下一模块**

- GPT-4o 程序生成；
- Cutie 物体跟踪；
- 双目 DINOv2 关键点跟踪；
- nvblox 地图生成。

### 4.4 在线跟踪：获得当前关键点和物体状态

该部分包含两种不同的跟踪，不应混淆。

#### DINOv2关键点跟踪

**目的**：持续更新 `K0...Kn` 当前的三维位置。

代码：

```text
src/rekpiper_perception/scripts/dual_dino_tracker_node.py
```

工作方式：

- 未抓取物体：在两台相机中进行 DINOv2 特征匹配并恢复三维位置；
- 已抓取物体：把关键点保存到夹爪局部坐标系，再通过当前夹爪 TF 转回 `base_link`；
- 视觉观测和刚体预测差异过大时，输出无效状态并阻止规划。

输出：

```text
/rekpiper/tracking/keypoints
```

#### Cutie物体掩码跟踪

**目的**：跟踪物体在两台相机图像中的二维区域。

代码：

```text
src/rekpiper_perception/scripts/multicamera_object_tracker_node.py
```

Cutie 输出二维 mask 和置信度，不直接输出 ReKep 三维关键点。它主要服务于：

- 动态物体地图排除；
- 物体身份和生命周期维护；
- 已抓物体的碰撞云；
- 丢失与重新获取判断。

**下一模块**

- 当前关键点进入 `closed_loop_node.py`；
- 动态 mask 进入 nvblox；
- 物体状态进入对象注册器、AnyGrasp 和闭环执行器。

### 4.5 nvblox：生成安全距离场

**目的**

建立当前工作空间的障碍物距离场，回答“某个位置距离障碍物还有多远”。

**输入**

- 双相机 RGB-D；
- 相机外参；
- cuRobo 生成的机械臂 mask；
- Cutie 生成的动态物体 mask。

**方法与原理**

1. 将过滤后的深度融合成 TSDF；
2. 从 TSDF 更新 ESDF；
3. 通过 `/rekpiper/mapping/query_sdf` 查询距离；
4. 生成与 ReKep 求解器工作空间一致的规则 SDF 网格。

代码：

```text
src/rekpiper_mapping/scripts/nvblox_mapping_node.py
src/rekpiper_mapping/scripts/sdf_grid_snapshot_node.py
```

SDF 约定：

- 负数：自由空间；
- 0：障碍物表面；
- 正数：障碍物内部；
- 未观察区域：按零净空/不可安全穿越处理。

**输出**

```text
/rekpiper/mapping/sdf_grid
/rekpiper/mapping/safe_status
```

`SDFGrid` 包含距离数组、观测标记、空间边界、分辨率和 `map_generation_uuid`。

**下一模块**

- SubgoalSolver；
- PathSolver；
- 全机械臂轨迹审计；
- AnyGrasp 抓取候选审计。

### 4.6 GPT-4o：从语言到ReKep约束程序

**目的**

把自然语言任务转换成求解器能够计算的分阶段空间约束。

**输入**

- 用户任务指令；
- `SceneSnapshot`；
- 标有 `K0...Kn` 的候选图；
- Piper 能力描述。

**服务**

```text
/rekpiper/program/generate
/rekpiper/program/approve
```

**代码**

```text
src/rekpiper_planning/scripts/program_server_node.py
src/rekpiper_planning/src/rekpiper_planning/vlm_program_generator.py
src/rekpiper_planning/src/rekpiper_planning/official_program.py
```

GPT-4o 不直接输出机器人轨迹，而是生成：

- 任务阶段；
- 每阶段子目标约束；
- 每阶段路径约束；
- 抓取关键点；
- 释放关键点。

约束函数接收当前末端和关键点状态，返回约束违反程度。违反程度小于配置容差才算满足。

程序还要经过：

- 语法解析；
- AST 白名单检查；
- 数值抽样检查；
- 文件哈希计算；
- 人工或签名审批。

**输出**

```text
/rekpiper/program/current
```

消息类型为 `ReKepProgram`，主要包含：

- `session_id`；
- `snapshot_id`；
- `program_sha256`；
- 阶段数；
- 抓取和释放关键点；
- `approved`。

**下一模块**

```text
src/rekpiper_execution/scripts/closed_loop_node.py
```

### 4.7 ClosedLoopNode：组合当前世界状态

**目的**

在每个规划周期中，把程序、视觉、地图、机器人和物体生命周期组合成一次完整的求解请求。

**输入**

- 已批准的 `ReKepProgram`；
- 对应的 `SceneSnapshot`；
- 当前双目跟踪关键点；
- 当前 SDF；
- 当前关节角；
- 当前末端 TF；
- 物体注册和附着状态；
- AnyGrasp 候选；
- 安全地图状态。

代码：

```text
src/rekpiper_execution/scripts/closed_loop_node.py
  ClosedLoopNode._tick()
```

求解前必须满足：

- 数据没有超时；
- 关键点全部有效；
- `snapshot_id` 一致；
- `program_sha256` 未被修改；
- 地图状态为 `READY`；
- SDF 和安全状态的 `map_generation_uuid` 一致。

任何一项失败都会安全暂停，而不是继续使用旧数据。

**输出给下一模块的数据**

`RealtimePlanningRequest`，包含末端位姿、关节角、关键点、刚体组、当前阶段、SDF、碰撞点和抓取状态。抓取阶段先等待已绑定的 AnyGrasp 候选，再传入 `grasp_target_pose`，直接作为 PathSolver 终点；该阶段跳过 SubgoalSolver。详见 [关键点抓取阶段说明](KEYPOINT_ANYGRASP_STAGE.md)。

### 4.8 SubgoalSolver：下一步应该到哪里

**目的**

求解当前任务阶段中满足语义约束的目标末端位姿。

官方代码：

```text
third_party/ReKep/subgoal_solver.py
  SubgoalSolver.solve()
```

适配调用：

```text
src/rekpiper_planning/src/rekpiper_planning/realtime_planner.py
  PersistentOfficialPlanner.solve()
```

**主要输入**

```text
当前末端7D位姿
当前关键点坐标
关键点可移动掩码
子目标约束
路径约束
SDF距离网格
夹爪及已抓物体碰撞点
是否为抓取阶段
当前关节角
```

`keypoints[0]` 在官方求解接口中是末端位置。未抓取物体的关键点不随候选末端运动；被抓物体同一刚体组的关键点会随候选末端刚性变换。

**原理**

- 优化变量是下一目标末端位姿；
- 代价包含任务约束、碰撞、可达性和抓取接近等项；
- 第一次求解使用 Dual Annealing 进行全局搜索；
- 后续闭环重规划使用上次结果进行 SLSQP 局部优化。

**输出**

- `subgoal_pose`：当前阶段的语义目标末端位姿；
- `debug_dict`：代价、约束和求解诊断信息。

返回一个 7D 位姿不等于求解已经被接受。结果仍须经过约束复核、IK、碰撞审计和真实执行反馈。

**下一模块**

```text
PathSolver
```

### 4.9 PathSolver：应该怎样到达目标

**目的**

从当前末端位姿规划一条满足路径约束、远离障碍物的笛卡尔路径。

代码：

```text
third_party/ReKep/path_solver.py
  PathSolver.solve()
```

**输入**

- 当前末端位姿；
- SubgoalSolver 输出的目标位姿；
- 当前关键点和可移动掩码；
- 路径约束；
- SDF；
- 碰撞点；
- 当前关节角。

**原理**

1. 优化若干中间控制位姿；
2. 计算路径约束和碰撞代价；
3. 使用样条插值得到密集笛卡尔路径；
4. 对各路径点运行 Piper IK；
5. 检查关节跳变和连续性。

**输出**

- 语义子目标位姿；
- 实际执行目标位姿；
- 密集笛卡尔路径；
- Piper 关节路径；
- 子目标和路径约束值；
- 求解延迟。

当前 `system.launch` 使用 `paper_real` 求解配置。它在官方 ReKep 几何求解基础上增加真实机械臂一致性、桌面净空和全机械臂安全相关代价。

**下一模块**

- Piper IK 结果复核；
- 全机械臂轨迹碰撞审计；
- 短时域轨迹生成。

### 4.10 轨迹审计与ReKepHorizon

**目的**

在规划结果进入硬件前，确认整条机械臂和已抓物体在每个路径点都满足安全要求。

代码：

```text
src/rekpiper_planning/src/rekpiper_planning/trajectory_audit.py
src/rekpiper_execution/scripts/closed_loop_node.py
```

主要检查：

- 关节上下限；
- 起点与当前关节反馈是否一致；
- 速度、加速度和 jerk；
- 全机械臂与 ESDF 的距离；
- 已抓物体与环境的距离；
- 最小净空；
- 地图版本是否仍然一致。

**输出**

```text
/rekpiper/planning/horizon
```

`ReKepHorizon` 包含：

- 语义目标；
- 实际执行目标；
- 预测笛卡尔路径；
- 授权关节前缀；
- 约束值；
- 规划延迟；
- 地图、场景和程序绑定；
- `valid` 和 `authorized`。

每次只授权未来约 `0.1 s` 的关节轨迹：

- Shadow 模式只发布 `shadow_validated` 预览；
- Autonomous 模式满足全部门控后才发送短轨迹。

**下一模块**

```text
Piper trajectory bridge
```

### 4.11 AnyGrasp：决定具体怎样夹住目标物体

**目的**

在 ReKep 已经决定“抓哪个关键点所属物体”之后，为 Piper 生成实际可执行的夹爪位姿。

代码：

```text
src/rekpiper_grasp/scripts/keypoint_anygrasp_node.py
src/rekpiper_grasp/src/rekpiper_grasp/anygrasp_adapter.py
src/rekpiper_grasp/src/rekpiper_grasp/grasp_geometry.py
```

**输入**

- 已批准程序中的抓取关键点；
- 当前 ReKep 抓取阶段和 Horizon；
- 目标刚体组和唯一物体 UUID；
- RS1 工作空间点云；
- 当前关键点；
- 当前关节角；
- SDF 和安全地图版本。

**方法与原理**

1. 从 RS1 工作空间点云及当前实例 mask 点云确定抓取关键点所属物体区域；
2. AnyGrasp 筛选基坐标系向下 30° 内的抓取姿态；
3. 将 AnyGrasp 相机坐标姿态转换成 Piper TCP 姿态；
4. 检查候选是否接近指定关键点；
5. 检查接触点、夹爪宽度、IK、机器人碰撞和 ESDF 净空；
6. 优先选择 TCP 到抓取关键点距离最近的安全候选。

**输出**

```text
/rekpiper/grasp/candidates
/rekpiper/grasp/selected
```

候选在路径规划前生成并锁定，转换后的 Piper TCP 直接设为抓取阶段子目标。先验证 OPEN 的实时开度，再开始执行规划路径；实测 TCP 位置和朝向连续满足到达条件后，才允许 CLOSE。

**下一模块**

- 抓取接近轨迹；
- 夹爪动作；
- 物体附着验证。

### 4.12 Piper执行与反馈闭环

**目的**

将经过授权的短关节轨迹发送到 Piper，并用真实关节反馈验证执行过程。

主要输入接口：

```text
/manipulator_controller/follow_joint_trajectory
```

轨迹桥再次检查：

- 硬件是否授权；
- 地图版本是否一致；
- 关节和机械臂状态是否新鲜；
- 轨迹起点是否匹配；
- 关节限制和跟踪误差。

桥接输出：

```text
/joint_ctrl_single
```

Piper 驱动最终调用：

```python
self.piper.JointCtrl(joint_0, joint_1, joint_2,
                     joint_3, joint_4, joint_5)
```

对应代码：

```text
src/piper/scripts/piper_ctrl_single_node.py
```

新的关节状态、末端 TF、图像、关键点和地图重新进入 ClosedLoopNode：

```text
执行0.1秒
→ 获取新反馈
→ 更新关键点和地图
→ 重新求解
→ 再执行0.1秒
```

这才构成真实 ReKep 闭环。

## 5. 抓取与释放阶段

抓取和释放不是普通的路径阶段，它们需要物理证据。

代码入口：

```text
src/rekpiper_execution/scripts/closed_loop_node.py
  ClosedLoopNode._stage_event()
```

### 5.1 抓取

```text
验证AnyGrasp候选与ReKep关键点绑定
→ 最近安全候选TCP设为抓取阶段子目标
→ PathSolver规划并审计路径
→ OPEN并确认实时开度
→ 保持张开执行接近路径
→ 实测位置和朝向连续满足到达条件
→ 夹爪闭合
→ 验证稳定接触
→ 执行约20mm验证运动
→ 验证物体与夹爪刚性运动一致
→ 标记物体为ATTACHED
→ 将物体从静态地图中排除
→ 重建地图
→ 进入下一阶段
```

### 5.2 释放

```text
确认存在唯一ATTACHED物体
→ 打开夹爪
→ 执行短距离撤离
→ 验证物体不再跟随夹爪
→ 标记物体为FREE_TRACKED
→ 将物体重新加入静态地图
→ 重建地图
→ 进入下一阶段
```

Shadow 模式不能提供真实接触、附着和释放证据。因此普通运动阶段可以预览，但到抓取或释放事件时会停止并报告：

```text
shadow_stage_event_requires_physical_lifecycle
```

## 6. 组件角色

这些组件不是六个依次串行执行的模型，而是分布在三个功能支路中。

| 支路 | 组件 | 输入 | 主要输出 | 服务对象 |
|---|---|---|---|---|
| 场景理解 | SAM | RGB图像 | 物体实例mask | DINOv2、对象注册 |
| 场景理解 | DINOv2 | RGB、mask、点云 | 关键点候选、在线关键点、身份特征 | GPT-4o、规划器、跟踪器 |
| 物体维护 | Cutie | RGB和初始物体mask | 在线二维物体mask | nvblox、对象生命周期 |
| 安全地图 | cuRobo | URDF、关节状态、相机参数 | 机械臂mask | nvblox |
| 安全地图 | nvblox | 双相机深度和过滤mask | TSDF、ESDF、SDFGrid | ReKep求解和碰撞审计 |
| 抓取执行 | AnyGrasp | 目标物体局部点云 | 抓取候选姿态 | 抓取执行器 |

ReKep 的 `SubgoalSolver` 和 `PathSolver` 是数值优化器，不是需要单独权重的视觉神经网络。

## 7. 操作者运行顺序

完整的控制顺序如下：

1. 配置 `REKPIPER_ROOT`、模型目录、第三方代码目录和数据目录。
2. 运行依赖和部署预检。
3. 启动 `start_rekpiper.py --mode shadow`。
4. 确认双相机、点云、TF 和关节状态有效。
5. 向 `/rekpiper/perception/task_request` 发布任务文本。
6. 等待有效且锁定的 `SceneSnapshot`。
7. 使用同一条任务指令调用 `/rekpiper/program/generate`。
8. 检查生成的阶段、约束和关键点绑定。
9. 使用返回的 `session_id` 和 `program_sha256` 调用 `/rekpiper/program/approve`。
10. 等待双目关键点、对象注册、安全地图和 SDF 全部有效。
11. 调用 `/rekpiper/execution/arm`。
12. 向 `/rekpiper/execution/execute_rekep` 发送 Action 目标。
13. 在 Shadow 模式检查每个 `ReKepHorizon`、约束、IK、碰撞和时延。
14. 完成标定和所有验收后，再进入 Autonomous 模式。
15. 在 Autonomous 中逐步验证短轨迹、抓取、附着、地图重建和释放。

当前仓库没有一个上层任务客户端自动串联第 5—12 步。`task_request` 在非测试代码中只有订阅者，没有内部发布者。因此当前需要人工调用 ROS 话题、服务和 Action，或者后续增加一个只负责顺序编排的轻量客户端。

## 8. 当前状态与下一目标

当前仓库仍明确处于验收前状态：

- `third_party/VENDOR.lock.yaml` 中 `approved: false`；
- 各第三方组件的 commit 和资源哈希仍为 `SITE_REQUIRED`；
- 两台相机外参状态为 `UNCALIBRATED`；
- `paper_real_solver_candidate.yaml` 中 `approved: false`；
- 相机、Piper 工作空间、夹爪基线、安全地图和急停验收均为 `false`；
- 默认关闭硬件命令。

因此当前最准确的阶段目标是：

> 先打通真实数据层和 Shadow 规划层；在依赖、标定、安全与性能证据全部验收前，不把节点启动、离线测试、7D 位姿返回或 Horizon 预览称为真机 ReKep 已跑通。

推荐按以下顺序推进：

1. **冻结运行依赖与模型版本**：补全并验证 VENDOR lock、模型哈希、CUDA 和许可证。
2. **完成双相机标定**：获得可接受的外参和 TF 证据。
3. **打通实时数据**：验证 RGB-D、点云、关节、TF 的时间和坐标一致性。
4. **打通感知跟踪**：验证快照、关键点、物体 UUID、Cutie mask 和双目 DINO 跟踪。
5. **打通安全地图**：验证机械臂排除、动态物体排除、ESDF 和地图代际绑定。
6. **打通约束程序**：验证 GPT-4o 程序生成、数值检查和人工审批。
7. **打通 Shadow 闭环**：持续生成有效 Horizon，并验证回退和暂停机制。
8. **验收 paper_real**：使用回放和现场数据确定权重与性能边界。
9. **分阶段验收硬件**：先保持、再单段短轨迹、再抓取和释放。
10. **完成任务级验收**：以 `ExecuteReKepResult.success=true`、全部阶段完成和完整审计证据为最终标准。

## 9. 最终成功判据

只有同时满足以下条件，才能称为“ReKep 已跑通”：

- 输入来自真实且已标定的相机、TF 和 Piper 反馈；
- 关键点、对象身份和地图在执行期间保持有效；
- GPT-4o 约束程序经过审查和审批；
- SubgoalSolver 和 PathSolver 的约束结果被下游安全检查接受；
- IK、关节限制和全机械臂碰撞审计通过；
- 每段硬件运动具有明确授权并获得反馈；
- 抓取具有稳定接触和附着证据；
- 释放具有脱离证据；
- 丢失跟踪、地图过期或约束失败时能够暂停或回退；
- `ExecuteReKepResult` 返回成功，所有阶段完成；
- 日志、回放和现场验收材料能够复现这一结果。
