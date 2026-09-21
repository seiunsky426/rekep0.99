# M5 连续跟踪与物体生命周期

2026-09-14：M5.1–M5.4 手动观测链路已跑通，用户完成红色圆柱移动与遮挡试验，视频及记录已核对。RS1 独立完成掩膜、关键点跟踪和对象身份维护；RS3 仅提供点云与辅助观察。夹爪判据及附着几何（M5.5/M5.6）按用户要求留到后续真机执行验证。

## 当前结果

- 本次 M4 快照 `snap-1789351896704755783-a6e79b5310e6`，选中桌面 7 个分割区域，全部注册成功，每组一个固定 UUID。分割区域数不等于物理物体数。
- 本轮跟踪 M4 原编号 K5–K10：红色圆柱 K5、绿色方块 K6、粉色方块 K7、黄色圆盘 K8/K9、蓝色方块 K10。其余选中区域没有 M4 候选关键点，仍维护对象状态。
- 用户确认移动/遮挡试验完成。试验窗口共 571 个跟踪输出，约 4.99 Hz，采集到接收延迟中位数 0.527 s、P95 0.649 s。
- K5 对应红色圆柱，初末稳定位置估计相差 0.288 m。视频中关键点随圆柱移动，在遮挡期间失效并断开轨迹线，重新可见后回到同一圆柱。该数值为相机估计位移，没有独立尺量真值。
- 红色圆柱的对象状态发生 `FREE_TRACKED → LOST → FREE_TRACKED`，最长连续 LOST 约 4.42 s。其 UUID 始终为 `7b4c86f0-0045-4fa0-971a-ef7cd2947b33`；本轮全部 7 个对象 UUID 均未变化。
- K5 有效 540/571 帧，主要无效区间对应遮挡；其他 5 个静止关键点各 570/571 帧有效，初末稳定位置差 1.9–6.8 mm。这是当前静止场景重复性，不能替代绝对精度。
- 实际 ROS 连接检查：M5 对 RS3 的订阅为零，`/rekpiper/objects/rs3/*` 输出为零。
- RS1 彩色轨迹窗口与 RViz 已打开。三维背景保留 RS1+RS3 的真实 RGB；关键点轨迹来自 RS1。

## 现场验收范围与限制

本次通过一个物体的手动移动、遮挡失效、恢复及身份保持验证，其余物体同时保持观测。不是多物体同时快速移动的鲁棒性验收，也未达到原规划 20 Hz 的频率目标。

遮挡边缘仍出现短暂三维跳点，采集/处理延迟约半秒；不要据此宣称遮挡瞬间定位准确。约第 91.67 秒有一帧所有关键点因未获得可用匹配而标无效，下一帧恢复，对象 UUID 不变。记录保留这些无效状态，未插值补成观测。夹爪/附着、精密控制和绝对外参精度均未在本次验证。

![K5 实测轨迹及可见性](../runtime/evidence/m5_20260914T020831Z/manual_trial_trajectory.png)

![遮挡后同一 K5 在新位置恢复](../runtime/evidence/m5_20260914T020831Z/trajectories/review_75s.png)

## 已处理的阻塞

1. 观测 ROS master 停止：在 `http://127.0.0.1:11321` 恢复相机和 M3。沿用 9 月 11 日 18 点在线内参 PARK 外参、X 上限 0.75 m；RS3 深度曝光 1000 μs、gain 16、laser 300。
2. 小而细的 SAM 区域缩到 DINO 16×16 网格后，最近邻采样可能全空：改用面积覆盖权重汇聚 patch 特征，保留非空区域，空掩膜仍拒绝。
3. M4 推理有延迟：M5 缓存采集时刻的 RS1 RGB-D，按快照时间选择参考，拒绝无对应参考帧的初始化。本轮参考归档且注册完成后停止 M4 推理，冻结本次编号。

## 入口与数据

- [手动观测 launch](../src/rekpiper_perception/launch/manual_tracking.launch)：不启动夹爪、附着监测或机械臂控制。
- [轨迹记录节点](../src/rekpiper_perception/scripts/keypoint_trajectory_node.py)：发布 RGB 叠加图、RViz MarkerArray；记录 CSV 与对象状态 JSONL，失效时断开轨迹线。
- 运行目录：`runtime/evidence/m5_20260914T020831Z/`；`reference.json` 和 `registration.json` 保存本次快照/分组/UUID，`manual_trial_summary.json` 保存固定试验窗口指标，`trajectories/` 保存现场 CSV、对象状态 JSONL、`manual_trial.avi` 及逐帧时间。视频为恒定 10 fps 封装，核对精确时刻使用 `manual_video_frames.csv`。
- 轨迹话题：`/rekpiper/tracking/keypoints`、`/rekpiper/tracking/trajectory_image`、`/rekpiper/tracking/trajectory_markers`。
- `manual_tracking.launch` 中 seed_bounds 仅用于初始选择桌面目标，不改变 M3 工作区，也不限制已注册物体后续移动。

252 项 Catkin 测试通过；仓库与 launch 路径审计通过。现场视频/轨迹是本次手动观测试验证据，离线测试单独列示；二者都不代表夹爪验收或标定绝对精度通过。当前跟踪输出 `motion_allowed=false`。
