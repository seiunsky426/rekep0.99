# M8 连续两级规划与 AnyGrasp 抓放验收

## 当前交付与证据边界

本次实现保留三个阶段及当前程序的 K10/K9 公式，不重新上传图像、不重新签署程序、不使能、不回零、不发送机械臂或夹爪命令。新增算法及执行代码必须重新进入现场发布验收，不能复用旧源代码/配置指纹的批准。

| 已验证项 | 本次结果 |
| --- | --- |
| 原归档失败 | 81 点路径，原始索引 1→2 跳变 0.444239 rad，最大跳变 1.664084 rad，已复现 |
| 公共连续 IK，0.10 rad | 仍拒绝该路径，失败索引 8；不会将局部失败后的估计当作轨迹 |
| 公共连续 IK，0.12 rad 诊断 | 仍拒绝该路径，失败索引 10；姿态放宽不能掩盖关节分支跳变 |
| 原生 OMPL 隔离回退 | 124 点，最大关节步长 0.0198823 rad；仅检查机器人自碰撞，未通过环境/任务验收 |
| 新双相机帧 | RS1、RS3 各 450 组同步 RGB/depth/CameraInfo 帧 |
| 实际 AnyGrasp SDK | 本地 RS1 全场景冒烟推理约 1.233 s，返回 10 个候选；不是绑定蓝色方块的任务候选 |
| 当前正式输入阻塞 | 无 `/rekpiper/objects/tracked_clouds` 与正式 `/rekpiper/mapping/sdf_grid` 数据；实际 joint2/joint3 超出建模边界 |

证据目录：`runtime/evidence/m8_upgrade_replay_20260914/`、`runtime/evidence/m8_upgrade_live_20260914/`。其所有候选均标记 `motion_allowed=false`。完整三阶段实机成功仍未验收。

## 改动与接口

- `PiperURDFIKSolver.validate_pose_sequence` 统一使用公共连续 IK：100 次求解预算，最多 8 个确定性初值、3 层插点细化，逐边核对 FK。原残差扫描改名为 `diagnose_pose_sequence`，仅作对照。
- `RealtimePlanningRequest` 增加可选 `joint_validity_fn`、`joint_fallback_fn`、`solver_call_timeouts`；`RealtimePlanningResult` 增加 `diagnostics`。原格式保持可用，无验证回调时不能启用回退。
- PathSolver 优先，连续 IK/状态检查失败或 PathSolver 超时后才回退。RRTConnect 每次最多 2 s、最多 3 次，只接受精确目标解；每条边以最大关节间隔 0.02 rad 检查任务约束、全臂自碰撞、环境及持物几何。
- 原生 `librekpiper_motion_backend.so` 使用本机 OMPL/MoveIt，仅提供数值计算，没有 ROS 运动接口。URDF 碰撞模型仅放行直接相邻链接触；声明了碰撞模型却无法加载几何时拒绝初始化。
- M8 工作进程可取消，总上限 60 s；子目标单次上限 30 s，PathSolver 20 s，给回退留出时间。配置默认采样预算为子目标 200、路径 40，局部优化 maxiter=30；原 upstream 文件未修改。
- 保留正式位置 0.01 m、姿态 0.10 rad、关节跳变 0.35 rad。0.12 rad 只在诊断脚本中使用。
- 返回候选绑定程序、场景布局、对象身份和当前状态。小漂移仍需重新审核；程序/身份变化、关节超过 0.01 rad 或关键点超过 5 mm 的漂移拒绝旧结果。
- 所有实际发送均通过既有轨迹 Action，按不超过 100 ms 的前缀执行；每段复查最新状态、地图、FK、任务约束与碰撞。时间参数化后再次审核速度/加速度/jerk。执行期间不在 ROS 回调里运行长时两级优化；自由物体移动超过 5 mm 时暂停，恢复后重新规划。
- `GraspCandidate.msg` 新增 `joint_motion_cost`；候选按审核后的关节运动代价、距离、网络分数选择。需要重编译并重启相关消息消费者，不能混用新旧 ROS MD5。

Python 工作进程采用 Linux `fork`，子进程仅调用 CPU 数值规划和本地校验，不调用 ROS 或 CUDA。取消、超时和子进程异常均丢弃结果；已测试离线进程生命周期。实际冷启动/持续运行延迟仍必须现场测量，不能以这些功能测试声称达到 10 Hz 两级重规划。

## 三阶段事件与失败处置

1. **接近与抓取**：ReKep 接近位姿稳定到达后触发 AnyGrasp。输入区域来自实时对象掩膜点云，以 UUID 绑定；包围盒不再作为目标归属依据。候选必须通过完整局部路径审核。先按建议宽度预张开、核验开口（3 mm 容差），再接近、闭合、读取新反馈并进行抓后验证。
2. **移动**：只在 `ATTACHED` 后进入，沿抓后确认的物体—末端关系传播 K10，目标仍为 `K9+[0,0,0.10]`。全程包含持物几何。掉落、跟踪丢失或地图失效立即暂停。
3. **放置与释放**：目标仍为 `K9+[0,0,0.02]`。先验证实际支撑关系，再打开并核对开口，沿世界 +Z 撤离验证。释放验证期间物体不再被假定随 TCP 运动；只有生命周期确认其留在世界中并恢复自由跟踪才完成。

接触例外仅针对已绑定目标的两个指腹区域（5 mm 邻域），使用实际开口几何并检查邻近其他物体；手掌和整段夹爪不豁免，未知体素不豁免。抓后验证中的携带几何属于验证假设，不设置 `ATTACHED`；碰撞和视觉相对运动证据仍分别审核。

AnyGrasp 推理失败最多 3 次。闭合后若是否持物不确定，则停止并保持，不自动张开或盲退；人工处置并验证安全释放后才能重试。暂停/中止同时取消规划子进程、轨迹及夹爪 Action。

### 放置几何的签名输入

不接受通过普通 ROS 参数将放置检查设为通过。新增放置几何必须位于已验证发布包的 **workspace 签名工件** `payload.placement_geometry` 中，并绑定当前程序和对象：

```yaml
placement_geometry:
  accepted: false
  program_sha256: SITE_REQUIRED
  object_uuid: SITE_REQUIRED
  anchor_keypoint: 9
  uncertainty_m: 0.001  # 示例；必须来自真实测量，不能直接作为验收结果
```

验收者需证明对象点云包含可用于判断底面的完整几何，不能用只看见顶面的点云推断物体底面。算法要求几何不确定度不超过 3 mm；物体投影位于支撑面的凸包内，保留 2 mm 加不确定度的边界裕量；底面间隙加不确定度不超过 3 mm。2 cm 的 K10/K9 公式若产生悬空或穿透，则拒绝释放并输出冲突，不更改公式。

## 无运动验证与 RViz

从仓库根目录运行：

```bash
source setup.bash
unset DASHSCOPE_API_KEY
catkin_make -j2
catkin_make run_tests -j2
catkin_test_results build/test_results
python3 tools/check_repository.py --source-root "$PWD"
python3 tools/check_launch_paths.py --package-root "$PWD/src" --dump-shadow
```

归档回放（输出目录必须是新的，避免覆盖原始证据）：

```bash
python -u tools/verify_m8_upgrade.py \
  --evidence-directory runtime/evidence/m8_recompute_20260914T082436Z \
  --output-directory runtime/evidence/m8_upgrade_replay_new
```

只读实时输入与 SDK 冒烟检查（相机及只读反馈需已启动）：

```bash
timeout 90s python -u tools/check_m8_live_inputs.py \
  --output-directory runtime/evidence/m8_upgrade_live_new --sdk-smoke-only
```

该脚本不发布任何抓取候选；`sdk-smoke-only` 使用未指定目标的 RS1 新点云，只证明 SDK 实际可推理。目标绑定推理必须等待 M5 掩膜/UUID、当前程序与 M7 安全地图完整，不允许用该冒烟结果顶替。

显示隔离回退候选：

```bash
roslaunch rekpiper_planning m8_offline_preview.launch \
  preview_data:="$PWD/runtime/evidence/m8_upgrade_replay_20260914/self_collision_only_candidate.npz"
```

仅发布到 `/m8_preview` 的显示关节和隔离 TF，不连接硬件控制器。橙色路径标记为“只验证自碰撞、环境未验收”，不表示可执行。旧 121 点与新 81 点显示数据均兼容。

## 进入现场验收之前

- 修复实际关节零位/编码器与模型边界的一致性；不能将现场反馈替换成假定全零。
- 完成双相机外参、TCP、夹爪、安全地图、急停及新源代码/配置指纹的签名验收。
- 启动并验证 M5 对象掩膜、UUID、实时关键点，M7 正式安全 SDF；分别检查实际数据率、时间戳和完整性。
- 完成目标绑定 AnyGrasp 候选、全臂及持物环境审核。阶段2、3任何假设抓取关系的预览都必须标明假设。
- 单独取得运动授权后，依次验收接近、抓取验证、运输、支撑检查与释放。保存 Action 结果、对象生命周期、候选/轨迹、ROS bag、逐阶段耗时与失败原因。

最终标准是蓝色方块确实被抓起并留在黄色圆盘上、三个阶段完成且执行结果成功。本次没有达到这个物理验收标准，也没有自动使能或执行的启动命令。
