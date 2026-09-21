# M6：分步规划距离场

2026-09-14：**按当前分步实验范围，M6 已跑通。** 采用“请求新地图 → 清图重采 → 导出完整距离场 → 计算短段”的流程。机械臂运动、夹爪与附着验证留到后续真机实验。

## 本轮完成情况

| 子项 | 当前结果 |
|---|---|
| M6.1 机器人主体过滤 | 沿用真实关节反馈和两路 cuRobo 几何掩膜；未建模标定板、支架保留为障碍 |
| M6.2 工作区深度过滤 | 沿用已验证的米制深度、工作区和掩膜时间戳规则；只纳入请求之后采集的帧 |
| M6.3 TSDF/ESDF | 每次请求重新融合两路深度，采够至少 5 帧/相机后更新 ESDF；未知空间保持保守占用 |
| M6.4 清图与版本 | 每次清图生成新地图 UUID，旧栅格发布内容先失效；采集完成后冻结本批地图 |
| M6.5 规划数据出口 | 建图节点内部直接采样完整 58×68×48 栅格；标准 SDFGrid、NumPy 数组和 ReKep 距离查询均已接通 |

固定条件：9 月 11 日 18 点在线内参 PARK 外参；工作区 `[-0.1,-0.45,-0.1]` 至 `[0.75,0.55,0.6]` m；名义分辨率 0.015 m。栅格坐标为包含上下边界的 linspace，消费者根据边界和数组尺寸构造坐标轴。

## 现场结果

证据目录：`runtime/evidence/m6_step_20260914T034542Z/`。

| 请求 | 完整服务往返耗时 | RS1/RS3 新帧数 | 已观测比例 | 地图版本尾号 |
|---|---:|---:|---:|---|
| 1 | 2.665 s | 5 / 5 | 31.50% | -1 |
| 2 | 2.204 s | 5 / 5 | 31.91% | -2 |
| 3 | 2.856 s | 6 / 5 | 31.65% | -3 |

三次均低于 10 秒，输出均为 189,312 个距离值和对应 observed 标记；地图 UUID 不同，所有参与帧的时间晚于各自请求。**M5 全程未暂停，结束时 K5–K10 共 6 个关键点有效，7 个对象 UUID 全部保持。** CAN TX 为 0。

最终节点重启后的命令入口检查另取一份新图，耗时 3.153 秒，记录在 `capture_handoff/`。冻结完成后的状态提示为 `stepwise_snapshot_ready_for_preview`，不会把有意停止连续建图误报成采集故障。

![分步距离场与短段数值预览](../runtime/evidence/m6_step_20260914T034542Z/stepwise_result.png)

图中蓝色为已观测自由空间，黑色为未知，黄色为离线短段 IK 预览；机械臂未执行该轨迹。

短段数据接入验证采用当前关节记录、现有 URDF IK 默认误差阈值，以及第三份快照：向上 3 cm 的 7 个路点全部求解成功，最大位置残差约 0.354 mm。末端采样点全部已观测，地图中的最小点净空约 0.118 m；官方 ReKep `PathSolver._setup_sdf` 的查询结果与快照读取器一致。该数值是本图内部的点距离，未覆盖整个夹爪或机械臂，也不是独立物理净空测量。本次为直线采样＋IK 接线预览，完整任务约束优化在 M7/M8 推进。

- `capture_01/` 至 `capture_03/`：`sdf_grid.rosmsg`、`sdf_grid.npz`、`capture.json`。
- `before_live.json`、`after_capture.json`：M5 身份与只读关节记录。
- `preview_short_segment.py`、`short_segment_preview.json`、`short_segment_preview.npz`：可复现的短段计算与距离查询记录。
- 264 项 Catkin 测试通过，仓库及 launch 路径审计通过。新增测试覆盖重采、超时、并发请求、旧帧拒绝、ROS 字节数组及未知空间插值。

## 使用入口

当前会话已运行在 `stepwise_capture:=true`。请求一次新地图并保存结果：

```bash
cd /home/zhengsihze/Rekep_v1.0-main
source setup.bash
source runtime/evidence/m6_step_20260914T034542Z/session.env
rosrun rekpiper_planning capture_stepwise_sdf.py \
  --output-dir "runtime/evidence/m6_step_capture_$(date -u +%Y%m%dT%H%M%SZ)"
```

已有相机、TF 和只读关节反馈时，建图启动入口为：

```bash
roslaunch rekpiper_mapping safe_dual_mapping.launch \
  observation_only:=true stepwise_capture:=true mode:=shadow \
  workspace_bounds_min:='[-0.1,-0.45,-0.1]' \
  workspace_bounds_max:='[0.75,0.55,0.6]' \
  mapping_cameras_config:="$PWD/runtime/evidence/m6_step_20260914T034542Z/observation_cameras.yaml" \
  rs1_extrinsics:="$PWD/runtime/evidence/m3_18pts_20260912T083133Z/rs1_diagnostic_extrinsics.yaml" \
  rs3_extrinsics:="$PWD/runtime/evidence/m3_18pts_20260912T083133Z/rs3_diagnostic_extrinsics.yaml"
```

该入口复用独占建图启动器；当前正在运行时不重复启动。分步模式不启动旧的周期性 `sdf_grid_snapshot_node.py`。

服务 `/rekpiper/mapping/capture_sdf_snapshot` 使用空请求，返回 `success、message、grid`。`success=true` 表示本批距离数据完整可供数值规划；`grid.valid=false`、`motion_allowed=false` 保留真实运动未验证的状态。结果另发布到 `/rekpiper/mapping/diagnostic_sdf_grid`。同一时间只允许一次采集，重入返回 `capture_in_progress`；等待新深度超时返回具体原因并清空输出，不复用旧栅格。

规划代码通过 `PlanningSDFSnapshot(grid)` 获得 `sdf_voxels、observed、bounds_min、bounds_max、generation_uuid、stamp_s`。每个规划批次重新请求；固定快照保留采集时间，不伪装成实时数据。距离符号为自由空间负、表面零、障碍内部正；未知和越界查询按占用处理。

## 后续推进与卡点处理

下一步进入任务约束与短段轨迹计算。运动中的掩膜变化、整臂碰撞与反馈跟随、夹爪附着、全面精度及连续高频性能留到相应真机实验，不作为本轮 M6 的完成条件。

若某一步受阻，先暂停该步骤、报告原因并给出替代方案。按需采集若再次受算力争用影响，替代方案为暂时分时运行 M5 与建图；本次三次采集已通过，未启用此替代方案。

旧连续模式证据保留在 `runtime/evidence/m6_20260914T022728Z/`：当时 15 次输出仅 5 次完整。该问题已通过按需快照绕开，本轮没有声称连续高频模式已修复。
