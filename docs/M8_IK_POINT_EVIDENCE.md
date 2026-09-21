# M8 候选路径的逐点 IK 证据

`PiperURDFIKSolver.validate_pose_sequence()` 已调用 `validate_continuous_ik()`。
现有流程包含上一解作种子、最多 8 个局部种子、最多 3 层细分、0.35 rad
关节步长上限，以及关节插值后的 FK 位姿残差检查。本次只增加可选
`trace` 记录，不改变这些决策，也不修改 PathSolver 目标函数。

## 回放同一条归档红路径

```bash
source setup.bash
MPLCONFIGDIR=/tmp/rekpiper_matplotlib python tools/diagnose_m8_red_path.py \
  --evidence-directory runtime/evidence/m8_blue_cube_20260914T074834Z \
  --output-directory runtime/evidence/m8_red_path_new_replay
```

输入直接取 `m8_preview_data.npz` 的 `candidate_path_poses7`，校验预览文件、
URDF 的归档 SHA256，并核对归档 q0。输出目录必须不存在，避免覆盖证据。
起始关节来自该归档的零位假设，不能称为当前实测状态。

输出内容：

- `report.json`：输入和代码哈希、模型、预算、软件版本、每次种子/细分尝试。
- `waypoints.csv`：所有原始点的 xyz、xyzw 四元数、连续检查状态；被检查到的点含
  q、实际前一关节状态、种子、六关节 Δq、残差、限位余量及 Jacobian 诊断。
- `attempts.csv`：完整尝试记录，包括细分区间、选择/提交状态和 FK 边检查结果。
- `diagnostics.png`：路径及残差、关节差分、限位/奇异性诊断曲线。
- `continuous_ik_preview.npz`：原路径不变的 RViz 输入。

连续检查在首个失败原始点停止。后续点为 `not_reached`，连续检查列留空；
另有每个点都固定用 q0 的独立求解列，便于区分单点可达性与连续路径。
独立求解的相邻差分不是可执行关节路径。Piper 的有界关节直接求差，不做
`wrapToPi`。`selected` 的局部细分只有在整个原始点通过后才标为 `committed`。

## RViz

```bash
roslaunch rekpiper_planning m8_offline_preview.launch \
  preview_data:=$PWD/runtime/evidence/m8_red_path_new_replay/continuous_ik_preview.npz \
  robot_description_file:=$PWD/runtime/evidence/m6_step_20260914T034542Z/robot_description.urdf
```

建议在独立 ROS master 下使用。机器人固定显示归档 q0；失败结果不会驱动动画。

| 显示 | 含义 |
|---|---|
| 红线 | 同一条被拒绝的笛卡尔候选路径 |
| 绿点 | 已提交的连续 IK 前缀 |
| 黄点 | 首次失败原始点的首个整段尝试为关节跳变 |
| 紫点 | 首次失败原始点的首个整段尝试残差超限 |
| 灰点 | 连续检查尚未到达，不能当成连续可达 |
| 橙色半透明外圈 | 该诊断关节解接近限位或奇异性警示阈值 |

风险标记采用最小限位余量 <2°，或缩放 Jacobian 的最小奇异值 <0.001。
平移行按 0.5 m 缩放，中心差分步长 1e-6 rad；这些是显示指标，没有新增拒绝门槛。
FK 边检查遇到失败即停止，报告的最大误差仅覆盖实际检查过的样本。
本模块不替代碰撞、任务约束、时间参数化或实机跟随验收。

## 2026-09-16 原路径核验

原预览 SHA256 为 `5ac608e31b8027141c2ce13b1bf89ff9b1c97126ae3af28735520874e49725e0`。
当前连续检查通过原始索引 0–8；第 6、7 点经细分通过，第 9 点最终失败。
该点首个整段解的位姿误差为 2.682 mm / 0.07611 rad，满足当前 IK 容差，
但最大关节变化 0.89417 rad 超过 0.35 rad。最后失败细分区间为该原始边的
`[0.25, 0.375]`，8 个种子均因关节变化约 0.83234–0.83396 rad 被拒绝。

这证明当前有限预算的连续检查未通过，不证明不存在其他连续分支。
后续 111 点未进入连续检查；固定 q0 的独立求解为 78/121 成功。
原先升级回放读取的 `recompute_lowbudget_report_attempt_0_path.npz` 并非
逐元素相同的红色预览路径，不能混用其失败点。

用户随后要求重新感知并规划，新的场景证据与以上归档回放分别保存。
