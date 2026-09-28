# ReKpiper 全包审计：距 ReKep 论文真机效果的差距

审计日期：2026-09-24。范围：本仓库当前工作树的 13 个 Catkin 包、`third_party/ReKep`、配置、现有现场记录及最新人工监督工作台代码。工作树已有未提交修改；本文按**当前磁盘内容**判断，未修改这些实现。此次只做源码、配置和已有证据检查及无硬件测试；没有启动相机、CAN、机械臂或发送控制命令。历史记录只能证明当时的输入和版本，不能替代今天的现场验收。

## 一句话结论

**软件链路已搭建，冻结场景的抓取与运输达到离线规划通过；完整放置、可信现场几何、连续闭环和任何物理抓放均未通过。** 当前最先挡住正式真机执行的是相机至机械臂基座的独立精度复核及其发布批准；即使这一项通过，放置目标可达性、夹爪/持物证据、实时地图和闭环性能仍分别阻塞。不能用“13 包编译/398 项测试通过”或“某段路径审核通过”推断论文效果已复现。

## 1. 比较基准和证据口径

论文的目标是：RGB-D 与语言输入产生语义关键点及逐阶段 Python 关系约束；子目标与路径两级优化；在执行中持续追踪关键点、更新 ESDF、重规划并在约束失效时回退。论文报告后续局部求解约 **10 Hz**，多相机 DINOv2 关键点跟踪 **20 Hz**；真实系统的 ESDF 使用多相机深度并排除机械臂和持物。[论文正文 §3–4](https://arxiv.org/html/2409.01652#S3)、[附录 A.7–A.9](https://arxiv.org/html/2409.01652#A7)

官方公开仓库是 **OmniGibson 演示代码**，其 README 明确说真实实验的关键点跟踪、mask 跟踪和 SDF 重建代码并未公开；不能把本项目中 ROS、Piper、nvblox、Cutie 或 AnyGrasp 的接口存在当作“官方真机实现已运行”。[官方 README](https://github.com/huangwl18/ReKep/blob/main/README.md#real-world-deployment) 本仓库将官方代码固定在提交 `63c43fdba60354980258beaeb8a7d48e088e1e3e`，见 [`UPSTREAM.lock.yaml`](../third_party/UPSTREAM.lock.yaml) 和 [`third_party/ReKep`](../third_party/ReKep/)。官方的 `main.py` 是仿真执行入口；本项目用 [`closed_loop_node.py`](../src/rekpiper_execution/scripts/closed_loop_node.py)、[`realtime_planner.py`](../src/rekpiper_planning/src/rekpiper_planning/realtime_planner.py) 与 ROS/CAN 桥接真机。

本文把证据分为四层：**代码存在 → 无硬件测试通过 → 指定现场的只读/离线结果通过 → 物理闭环任务通过**。同一模块的前一层不能自动提升后一层。“完成论文效果”至少要在 Piper 上从自然语言和新场景开始，完成抓取、运输、放置、失败检测和扰动后重规划，并按多次独立试验报告成功率/时延。论文在不同硬件和任务上的成功率（自动约束总体 44.3%，每任务 10 次）只能当方法背景，不能拿来算 Piper 的完成百分比。[论文 Table 1](https://arxiv.org/html/2409.01652#S4.SS1)

## 2. 当前真实进度：沿数据流逐包对照

| 包/环节 | 已有实现与最高证据层 | 仍缺的真机验收 |
|---|---|---|
| `piper`, `piper_msgs`, `piper_description`：反馈/几何 | 驱动、消息、URDF、FK/IK 接口存在；M1 只读反馈和数值几何检查见 [`M1_PROGRESS.md`](M1_PROGRESS.md) | 真实关节零位与 URDF 限位的一致性、速度/急停/控制模式及多姿态物理 TCP 精度；历史 M8 曾出现实测 J2/J3 略越模型边界。 |
| `rekpiper_camera`, `rekpiper_calibration`：双 RGB-D/外参 | 两路同步 RGB-D、点图、诊断融合和相对/手眼变换链已实现；M3 观测功能有现场记录 | 生产 [`rs1_extrinsics.yaml`](../src/rekpiper_camera/config/active_fixed_camera_extrinsics/rs1_extrinsics.yaml) 与 [`rs3_extrinsics.yaml`](../src/rekpiper_camera/config/active_fixed_camera_extrinsics/rs3_extrinsics.yaml) 仍是 `UNCALIBRATED`、禁止 TF/精密操作；工作区配置 `UNVERIFIED`。9 月 21 日相对结果仍 `PENDING_REVIEW`。必须用独立基座已知点和双相机重叠目标验证，且明确 `base_link_T_arm_base`。 |
| `rekpiper_perception`：初始关键点/掩膜/跟踪 | RS1 SAM+DINOv2 场景快照、RS1 Cutie/DINOv2 跟踪与对象 UUID 在静态及一次手动移动/遮挡试验通过；[`M4_PROGRESS.md`](M4_PROGRESS.md)、[`M5_PROGRESS.md`](M5_PROGRESS.md) | 9 月 14 日实测约 **4.99 Hz**、延迟中位 **0.527 s**、P95 **0.649 s**，低于论文 20 Hz 跟踪。正式 [`online_tracking.launch`](../src/rekpiper_perception/launch/online_tracking.launch) 的跟踪、注册和 tracker 都配置 `[rs1]`；RS3 仅辅助点云/当前工作台快照，尚无论文式跨视角连续追踪及移动中鲁棒性验收。夹爪状态/附着基线未完成。 |
| `rekpiper_mapping`：距离场 | M6 三次按需清图重采，58×68×48 栅格分别约 **2.67/2.20/2.86 s**；可供数值预览，见 [`M6_PROGRESS.md`](M6_PROGRESS.md) | 当前 [`safe_mapping_acceptance.yaml`](../src/rekpiper_mapping/config/safe_mapping_acceptance.yaml) 五个现场门均为 `false`；该记录 `grid.valid=false`、`planning_safe=false`，不能用于执行。生产 [`system.launch`](../src/rekpiper_bringup/launch/system.launch) 对安全地图设置 `use_rs3=false`，与论文多相机 ESDF 不同；运动中机器人/持物剔除、地图更新率与未知区覆盖未验收。 |
| `rekpiper_planning`：程序/两级求解/IK | 官方固定版 `SubgoalSolver`、`PathSolver` 经 [`upstream.py`](../src/rekpiper_planning/src/rekpiper_planning/upstream.py) 接入；有 `official_exact` 对照、Piper 稠密 IK/碰撞审计和 `paper_real` 候选。9 月 23 日冻结场景抓取和运输路径分别通过离线审计 | Qwen 在线生成的通用关系约束未验收；当前人工工作台 VLM 只选蓝块/黄盘编号，四阶段约束由本地模板生成，属于固定任务演示。冻结场景放置目标仍不可行；`paper_real` 候选 [`approved: false`](../src/rekpiper_planning/config/paper_real_solver_candidate.yaml)。全链路 10 Hz 热启动无数据支持。 |
| `rekpiper_grasp`：AnyGrasp | 有 SDK 候选选择、TCP 绑定、几何/接触/ESDF 审核；9 月 23 日某冻结输入的 `grasp-2-roll180` 获完整离线抓取路径 | 早先现场输入 28 个候选中 **0** 个通过夹爪 ESDF/全臂审核；冻结场景通过不等于现场抓稳。夹爪开度/effort、抓稳、滑落与持物几何尚未独立验收。 |
| `rekpiper_execution`, `rekpiper_bringup`, `rekpiper_acceptance`, `rekpiper_msgs`：闭环/放行 | 程序哈希、签名、地图/跟踪/关节时效、短轨迹与控制桥及人工监督界面均有实现；默认 `shadow`、`allow_hardware_commands=false`。当前工作树的 398 项 Catkin 测试通过 | 尚无已批准发布包、现场性能文件或物理阶段完成记录。人工工作台 8 份会话日志中没有 `preview_ready`、`execute_clicked`、`stage_completed`；归档中 6 次规划失败为 `experiment_calibration_check_required`。当前未提交代码已允许无复核预览，尚无新的现场预览记录。[`supervised_workbench_node.py`](../src/rekpiper_execution/scripts/supervised_workbench_node.py) 在 `EXPERIMENTAL_PREVIEW_ONLY` 下仍无条件拒绝执行。 |

最新离线结果需按输入区分：9 月 22 日“假定全零关节”实验的 CAN 为 DOWN、没有实测关节，28 个 AnyGrasp 候选最终为 0，详见 [`grasp_esdf_zero_start_20260922/RESULT.md`](../runtime/evidence/grasp_esdf_zero_start_20260922/RESULT.md)；9 月 23 日另一个**冻结场景**通过抓取和恢复后的运输审核，但放置三次搜索/修正仍失败，原正常运行耗时 **618.361 s**，没有任何硬件命令，见 [`grasp_timing_20260923/RESULT.md`](../runtime/evidence/grasp_timing_20260923/RESULT.md)。不能将这两组结果拼成一次完整任务通过。

## 3. 卡点与缺陷，按解除顺序排序

### P0：现场几何和执行授权没有闭合

1. **M2 的“5/5 完成”是流程标记，不能当生产精度放行。** [`M2_PROGRESS.md`](M2_PROGRESS.md) 与 [`REKEP_DEPLOYMENT_WORKFLOW_BASELINE.md`](REKEP_DEPLOYMENT_WORKFLOW_BASELINE.md) 写已完成，但当前两份正式外参仍 `UNCALIBRATED`、`publish_tf_allowed=false`、`precision_operation_allowed=false`；9 月 21 日 stereo 来源为 `PENDING_REVIEW`。这是**文档状态与生产配置冲突**，容易误导操作。修正口径应为“候选变换链和观测显示完成，独立物理精度/生产发布未通过”。验收产物应包含新批次拟合、封存验证集误差、跨批漂移、变换方向和已知点实测，之后才审查 TF/执行发布。
2. **归档工作台会话停在复核门；当前执行入口还有硬阻断。** 归档会话的 6 次 `planning_failed` 均为 `experiment_calibration_check_required`；一次复核命令返回 `at_least_four_independent_known_points_required`，见 [`events.jsonl`](../runtime/supervised/36c640e8e9a747349392689e0f1c5b4c/events.jsonl)。当前未提交代码的 `plan()` 已使用 `binding(require_calibration=False)`，所以这些旧失败**不能证明当前预览仍卡在同一原因**，必须用当前版本只读复测。工作台文档称复核通过后可执行，但当前 [`execute()`](../src/rekpiper_execution/scripts/supervised_workbench_node.py) 对配置中的 `EXPERIMENTAL_PREVIEW_ONLY` 先无条件返回 `experimental_extrinsics_preview_only`，即使会话复核成功也不会走到轨迹下发。它是安全保护，同时也是**文档与可达流程不一致**；应明确预览与受控执行的审批迁移路径，不得只改状态字符串绕过复核。
3. **多项相互独立的正式放行条件仍为假。** [`recognition_workspace.yaml`](../src/rekpiper_camera/config/recognition_workspace.yaml) 为 `UNVERIFIED`；[`piper_gripper_baseline.yaml`](../src/rekpiper_perception/config/piper_gripper_baseline.yaml) 未标定；[`safe_mapping_acceptance.yaml`](../src/rekpiper_mapping/config/safe_mapping_acceptance.yaml) 未接受；[`VENDOR.lock.yaml`](../third_party/VENDOR.lock.yaml)、`paper_real`、[软件](../src/rekpiper_execution/config/runtime_performance_software_unapproved.yaml)/[硬件](../src/rekpiper_execution/config/runtime_performance_hardware_unapproved.yaml)性能文件均未批准。生产入口的发布包默认空，硬件命令默认关闭。这些是显式安全门，不能以单次离线 PASS 替代。

### P1：有离线路径，但固定抓放仍未闭合

4. **放置不是已解决的规划问题。** 9 月 23 日抓取路径检查 1,240 点、最小净空约 8.79 mm；运输恢复路径检查 641 点、最小净空约 9.79 mm，但放置三次目标/路径尝试共 154.733 s 均未满足原位置、竖直姿态和 IK 约束。要检查黄盘真实位姿、Piper 末端/物体附着变换、目标放置姿态、关节可达域及保持原任务约束的替代路径；先在冻结输入上形成可审核的 S1/S2/S3 全通过，再谈物理执行。有限搜索失败不能证明数学上绝对不可达。
5. **真实起点和完整环境覆盖尚不稳定。** 9 月 21 日实测 J2/J3 曾略越 URDF 限位；9 月 22 日全零起点只是用户假设，RS1 ESDF 对起点全臂有 708 个未知采样及已知净空不足。端点 IK 或 MoveIt 自碰撞通过仍不代表 121/202 个笛卡尔位姿的连续 IK、全路径环境碰撞和持物碰撞通过。先解决编码器零位/模型一致性及自掩膜、未知区，再以**同一时间的实测关节和 RGB-D**重新规划。
6. **夹爪和附着是物理任务的空白验收。** 当前基线 `calibrated: false`；M5.5/M5.6 尚未完成空夹/接触/抓稳/滑落判别与持物关键点、碰撞点传播。人工点击“已夹稳”只是监督工作流的一环，不等于自动抓稳检测。必须保留视频、夹爪开度/effort、物体相对运动和释放后支撑证据。

### P1：与论文闭环能力的实测差距

7. **实时性相差一个量级以上，尚无闭环性能证明。** 论文热启动求解约 10 Hz、跟踪 20 Hz；本项目已测 RS1 跟踪约 5 Hz、约半秒延迟，M6 按需地图重采约 2–3 s，冻结场景官方全局路径搜索约 83–93 s、正常全流程一次约 618 s。论文的约 1 s 首次全局优化、约 10 Hz 后续局部更新与这里的冻结输入/硬件/约束并非同一基准，不能直接当性能回归数值；但目前证据明显不支持 `system.launch` 写的 10 Hz 规划率及 0.1 s 短轨迹连续执行。需要在真实共负载下测输入年龄、map/tracker/solver P50/P95、deadline miss、执行跟随和安全停止。
8. **真实多视角连续感知/地图与论文配置不同。** 论文附录使用所有可用 RGB-D 相机追踪关键点及多视角 ESDF；这里正式跟踪只用 RS1，生产安全地图也设为 RS1。RS3 的融合点云和工作台静态 mask 增加观察角度，却不能自动修复 RS1 遮挡、0.5 s 滞后或不可见体素。先验证单相机降级条件；若目标是论文式抗遮挡反应，则增加 RS3 连续跟踪和多相机 ESDF，并用同一物体遮挡/扰动试验比较。
9. **通用语言到关系约束尚未成立。** 生产后端虽配置 Qwen `qwen3-vl-plus`，[`README.md`](../README.md) 的 M7 记录仍是离线 API/语法测试，未有在线生成质量与人工批准验收；人工工作台目前只让 VLM 选两个编号，后续用固定模板。需用多个新场景、新物体和扰动任务，核对生成的阶段数、关键点引用、原始关系约束满足值与真实结果；固定蓝块放黄盘成功也只是单任务里程碑。

### P2：记录与当前行为的同步问题

10. [`REKEP_END_TO_END_FLOW.md`](REKEP_END_TO_END_FLOW.md) 数据流仍写“GPT-4o 生成”，但当前生产配置为 Qwen；M4 旧记录描述首次稳定帧等待，而当前未提交的 [`task_perception_node.py`](../src/rekpiper_perception/scripts/task_perception_node.py) 已改变初始捕获/锁定流程。它们是**文档漂移或待现场复验的行为变化**，不能把历史 M4/M7 测量直接覆盖到当前工作树。更新现场说明前，应以当前版本重做只读捕获和状态转换验收。

## 4. 距“论文效果”还差多少

按两个目标分别判断，而不报没有统计基础的“完成百分比”：

| 目标 | 当前可证明的位置 | 剩余必过门 |
|---|---|---|
| **最低单臂固定任务**：蓝块抓起、移到黄盘、释放 | 冻结场景 **抓取 S1、运输 S2 离线通过，放置 S3 阻塞；物理 S1–S3 均未完成** | 独立标定与工作区验收 → 实测起点/地图与可行放置路径 → 夹爪/附着/释放反馈 → 人工监督的逐段真实运动与成功证据。至少四类独立验收尚缺。 |
| **论文式真机闭环**：新语言任务、多阶段、扰动恢复 | 官方求解核心和闭环状态机已接线，尚无被接受的整段物理闭环试验 | 在固定任务通过后，另需通用 VLM 关系约束、连续多视角追踪/地图、热启动实时规划、跨阶段回退和多次新场景/扰动试验。Piper 单臂也不能声称复现论文的双臂任务。 |

**当前实际卡点链**：`外参/基座与工作区未验收` → `现场实测状态与可查询地图未同时有效` → `抓放三阶段未全规划通过` → `夹爪/持物反馈无验收` → `签名与性能放行未完成` → `没有物理闭环成功率`。前一个通过不会自动解决后一个。

建议按如下验收顺序推进，每一步保存输入哈希、配置、时间戳、完整失败理由和人工/自动边界：

1. **几何和状态**：封存独立已知点，复核两路 `base_link_T_camera`、深度尺度、RS1/RS3 同物体中心、桌面和 Piper 零位/URDF；先形成可审查报告，再决定生产 TF。验收：独立误差及跨批稳定性满足仓库门槛，不再使用假定零位。
2. **冻结场景全规划**：同一份真实 RGB-D、实测关节、目标/支撑 mask、有效 ESDF，保存 AnyGrasp 候选、三阶段约束值、Subgoal/PathSolver 原始结果、稠密 IK、全臂/持物碰撞与时间参数；验收：S1/S2/S3 全通过且可重放，不靠改约束/降低净空凑 PASS。
3. **受控物理单步**：先空载短轨迹与停机/跟随，再预抓取、抓稳、运输、放置，每段新感知和新审核；验收：实际关节/TCP、夹爪、物体轨迹和停止记录闭合，错误时停住。需完成正式安全批准，不能通过更改预览状态绕过。
4. **闭环和泛化**：把追踪/地图/求解分进程性能测量与扰动回退纳入目标，随后在预先定义的多次独立试验中统计成功率与失败类别；验收口径分别报告固定任务、自动约束任务、扰动任务。

## 5. 本次验证和局限

- `python3 tools/check_repository.py --source-root "$PWD"`：`clean: true`。
- `source setup.bash && python3 tools/check_launch_paths.py --package-root "$PWD/src" --dump-shadow`：`valid: true`。未 source 的首次检查因 ROS 包路径不存在而失败，属于命令环境问题；在项目规定环境中复测通过。
- `source setup.bash && catkin_make run_tests` 后读取 `catkin_test_results build/test_results`：**398 tests，0 errors，0 failures，0 skipped**。这是当前工作树的软件测试结果；不包含相机到基座精度、GPU 共负载、CAN/夹爪实动和真实抓放。
- 人工工作台日志统计仅覆盖当前磁盘中的 8 份 `runtime/supervised/*/events.jsonl`；没有完成事件只能说明这些归档会话未证明完成，不能断言其他未归档实验从未发生。
- 本次没有重新采集现场数据或比较论文基准任务的同条件成功率；时间、空间误差及软件门槛引用各自日期的归档，不把旧记录称为当前在线状态。
