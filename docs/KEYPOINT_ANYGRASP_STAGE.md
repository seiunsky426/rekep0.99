# 关键点选抓取目标，AnyGrasp 给定抓取阶段位姿

本地适配流程（2026-09-15）：抓取阶段先生成并选择 AnyGrasp 候选，再运行路径求解。第一阶段为抓取阶段时，采用以下顺序。该流程不是上游仿真中“先退半个 grasp depth，再执行局部抓取”的复刻。

1. 已批准程序的 `grasp_keypoints[stage-1]` 选择关键点，解析到唯一 `FREE_TRACKED` 物体 UUID。
2. 闭环发布 `grasp_target_pending`，`authorized=false`、空轨迹；此时不启动路径规划。AnyGrasp 请求绑定程序、快照、地图、阶段、物体和尝试次数。
3. 默认仅输入 `/rekpiper/camera/rs1/points_recognition` 工作空间点云，在 RS1 光学坐标系推理。目标区域来自当前实例点云。将基坐标系向下方向 `[0,0,-1]` 转换到推理相机系，限制接近方向偏离向下不超过 30°；无合格候选就报告失败，不自动扩大角度。
4. 保留通过原有接触归属、宽度、IK、自碰撞、ESDF 和接近检查的候选，按 **Piper TCP 到关键点距离** 升序选择；同距离时才比较关节运动量、网络分数和 ID。距离上限 10 cm。同一次绑定只锁定第一个选定候选，关键点移动超过 5 mm 则暂停并要求重规划。
5. 用 `grasp_pose`（已转换的 Piper TCP，含 AnyGrasp depth 和夹爪轴约定）设置 `RealtimePlanningRequest.grasp_target_pose`。该抓取阶段不调用 SubgoalSolver，不做 5 cm 后退；`semantic_subgoal_pose` 与 `target_pose` 均为候选位姿。
6. 抓取阶段原来的单个“末端到关键点”子目标，由适配层实例化为“末端到检测器 TCP”约束；程序仍负责选择物体和阶段，程序内容与签名不变。解析器仍要求抓取阶段只有一个子目标且无路径约束，其他阶段的原始约束和 SubgoalSolver 保留。PathSolver 以候选的位置和朝向为固定终点；后续密集 IK、关节连续性、实际 FK 路径和碰撞检查仍必须通过。
7. 自主执行放行后，发送 OPEN 并验证结果和实时开度（误差不超过 3 mm），再执行路径。运动中的每个受限轨迹段都检查夹爪仍张开。规划与执行审计使用当前开度的夹爪几何；目标接触例外仅限原有终点附近的目标指尖接触，不能豁免手掌、其他物体、未知地图区域。
8. 实测 TCP 的位置误差不超过 3 mm、朝向误差不超过 0.03 rad（用户配置更严则采用更严值），且连续满足到达次数后，复核目标绑定、实测位姿及夹爪开度，再进入 `begin_grasp` 和 CLOSE。到达后不再执行第二条局部接近路径。
9. 继续原有稳定接触与附着验证；通过后才进入下一阶段。CLOSE 返回成功本身不代表抓取成功。

## 代码入口

- 推理与筛选：`src/rekpiper_grasp/scripts/keypoint_anygrasp_node.py`、`src/rekpiper_grasp/src/rekpiper_grasp/selection.py`。
- 固定抓取子目标与路径求解：`src/rekpiper_planning/src/rekpiper_planning/realtime_planner.py` 的 `grasp_target_constraints()`、`solve()`。
- 调度与夹爪时序：`src/rekpiper_execution/scripts/closed_loop_node.py` 的 `_request_grasp_target()`、`_prepare_grasp_target()`、`_ensure_grasp_open()`、`_require_grasp_arrival()`、`_tick()`、`_stage_event()`。

## 验证边界

本次完成代码与无硬件测试，不启动驱动、不开夹爪、不发送机械臂轨迹。Shadow 只生成规划结果，不能凭模拟回执推进真实附着状态。

候选生成端仍保留原有从当前关节状态到候选的直线密集 IK/碰撞预审；这是保守的候选筛选，可能淘汰经绕行才能到达的姿态。选中候选后仍须运行 PathSolver 并重新完整审计，不能把候选预审当成路径求解成功。

RS1 只限定 AnyGrasp 的输入点云；正式规划仍依赖当前跟踪、对象注册、有效 ESDF、已验收外参及执行放行。先前 RViz 中的离线候选和诊断外参不满足这些条件，也不是本次改动的真机抓取验收证据。
