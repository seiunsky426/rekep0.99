# PathSolver 自动日志与 RViz IK 预览

在工作区执行：

```bash
source setup.bash
roslaunch rekpiper_planning path_solver_inspector.launch
```

这个入口打开 RViz 和独立的预览机器人，订阅已有的 `/joint_states_single`。
它不启动驱动、相机、CAN 或规划任务。正常闭环规划调用
`PersistentReKepPlanner.solve()` 时，PathSolver 日志默认自动记录。
更新代码后需在下一次启动闭环规划节点时生效，工具不会重启现有节点。
因此打开界面本身不会产生一条新规划路径；没有历史日志时仍可使用当前末端 IK。

## RViz 操作

选择工具栏 **Interact**，点击场景内的蓝色按钮：

1. **Live EE to target**：用新鲜关节反馈和当前 URDF 的 FK 更新可拖动末端目标。
2. **IK target**：以当前关节角为种子求目标 IK。可先拖动三轴平移/旋转手柄，再点击按钮看关节变化。
3. **Path IK from live joints**：从点击时的关节反馈开始，对日志里的完整密集路径求连续 IK，包括当前姿态到路径起点的连接。
4. **Replay IK**：重播上一次成功预览。

初始目标自动取当前末端位姿。机器人动画旁显示六个关节的角度（度）以及相对本次初始关节角的变化量。
当前姿态原地求 IK 通常不会产生明显关节变化；拖动目标或求整条路径可观察变化。
动画使用固定的显示频率，不代表机械臂速度或可执行的时间参数轨迹。
单目标动画在起止关节之间插值，**只验证目标 IK**；整条路径模式才运行连续 IK 和 FK 插值检查。
手指固定显示为闭合，不参与六轴 IK。
此预览 RViz 使用 Mesa 软件渲染：本机测试中 NVIDIA OpenGL 渲染标记时曾崩溃，
启动文件将软件渲染设置限定在这个 RViz 进程内。

颜色及数据：

| 显示 | 含义 |
| --- | --- |
| 品红大球、START/C1…/GOAL 标签 | 全部优化控制点（含起终点） |
| 黄色球 | 全部样条输出采样点 |
| 青色小球、连续线 | 抽稀的密集插值点、完整路径线 |
| 红绿蓝短轴 | 对应末端位姿的完整 XYZ 朝向 |
| 代价曲线青线 / 绿线 | 每次评估的总代价 / 截至当次评估的最优代价 |
| 可选 `ALL dense poses` 显示项 | 完整密集末端位姿，不经过显示抽稀 |

密集点默认最多显示 35 个，始终保留起终点。此设置不削减日志和 IK 输入：

```bash
roslaunch rekpiper_planning path_solver_inspector.launch max_interpolation_points:=20
```

## 日志位置和内容

默认目录是 `$REKPIPER_DATA_ROOT/path_solver/`；标准 `setup.bash` 对应
`runtime/data/path_solver/`。每次 PathSolver 调用创建独立目录：

- `events.jsonl`：初始化控制点、每次目标函数评估的归一化变量、全部候选控制点
  `[x,y,z,qx,qy,qz,qw]`、分项代价/约束违反量/IK 诊断、优化器结果以及结束状态。
  每条事件立即 flush；包括优化器拒绝的候选和最后一次 debug 评估。
- `costs.csv`：评估编号、时间、总代价、当前最优值及各代价项。
- `summary.json`：坐标系、初始关节状态、配置、阶段/规划版本、最终全部控制点、
  代价计算采样点 `objective_sample_poses`、样条点 `spline_poses`、密集点 `dense_poses`。
  后续规划成功返回时，还包含关节轨迹及实际 FK 位姿；若走关节空间回退，
  `diagnostics.path_backend` 会注明回退后端，原 PathSolver 路径仍单独保留。
- `cost_trends.svg`：查看该日志时自动生成的总成本、最优成本与分项成本曲线。

`latest.json` 原子指向最近**完成记录**的一次调用，包括失败结果。
状态 `planner_returned` 仅表示规划函数返回；`planner_failed` 保存异常；
被外部终止的进程可能留下 `incomplete` 记录，且不会替换 latest 指针。
若 PathSolver 尚未开始（如前置状态检查或 SubgoalSolver 失败），不会创建本工具日志。
原始日志没有抽样/轮转，会持续占用磁盘，诊断后按需归档。
完整逐次记录也增加 I/O 和序列化开销，仍受原有规划超时约束。

横轴是**目标函数评估次数，不是优化迭代次数**。数值梯度采样、退火候选及最后一次
调试评估都会计数；原始成本不保证单调，最优成本曲线才单调不增。
官方后端计算的 `reset_reg_cost` 没有加入总代价，CSV 保留它，SVG 标为
`diagnostic only`；paper-real 后端则加入该项。记录器不修改权重或优化策略。
非有限数在 JSON 中记为 `null`，CSV 留空；不能把它当作零成本。

每次按钮 IK 的证据写到同一日志根目录下的 `ik_previews/<id>/`：

- `ik.json`：目标、反馈时间、种子、结果、连续 IK 尝试及失败索引、关节角/增量。
- `joints.csv`：逐点关节角及相对种子的增量，单位 rad。
- `joint_trends.svg`：六轴角度与变化趋势，单位度。

IK 失败时保留已验证前缀作为诊断证据，但不播放成成功的整条路径。

## 参数与限制

固定查看某次记录：

```bash
roslaunch rekpiper_planning path_solver_inspector.launch \
  run_directory:=/absolute/path/to/a/solve
```

使用自定义日志根目录或反馈话题：

```bash
roslaunch rekpiper_planning path_solver_inspector.launch \
  trace_directory:=/absolute/path/to/path_solver \
  joint_state_topic:=/joint_states_single
```

规划侧可在闭环节点设置私有参数 `path_trace_directory`，或在 Python 构造
`PersistentReKepPlanner(..., path_trace_directory='/path')`；默认自动记录，
显式传入空字符串可关闭。查看器应使用同一目录。
`robot_description_file`、`base_frame`、`tip_frame` 必须匹配实际规划模型和日志。
默认使用仓库 Piper URDF、`base_link`、`rekep_tcp`；本工具不自动变换不匹配的日志坐标系。

按钮也提供相同的 `std_srvs/Trigger` 服务。例如：

```bash
rosservice call /path_solver_preview/inspector/solve_path
rostopic echo /path_solver_preview/inspector/status
```

服务返回的是请求是否入队，求解结果应看 status 和 `ik.json`。
反馈必须包含六个唯一关节名、有限且在 URDF 限位内的角度，以及最近 0.5 秒内的时间戳；
接收时间同样检查 0.5 秒新鲜度，拒绝零时间戳、未来时间和过期状态。

所有发布消息和 TF 位于 `/path_solver_preview/` 下；预览节点没有机械臂命令发布接口。
IK 成功不代表碰撞、ESDF、标定或执行授权通过。官方 `third_party/ReKep` 快照保持原样；
记录器只在单次求解中替换函数的局部全局查找表，不修改进程级模块或重算目标函数。
运行日志和本机测试截图遵循仓库约定放在 Git 忽略的 `runtime/`，代码、配置、测试与本文档提交 Git。
