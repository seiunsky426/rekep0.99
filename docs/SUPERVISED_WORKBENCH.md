# 人工监督抓放工作台

入口：

```bash
cd /home/zhengsihze/Rekep_v1.0-main
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch rekpiper_bringup supervised_workbench.launch
```

启动入口只打开会话服务与集成窗口；CAN、机械臂使能、双相机和 SAM/DINOv2 感知均由窗口按钮分别启动。单窗口内 RS1 可在实时图与编号快照间切换，RS3 保持实时图，第三画面嵌入 RViz 的双相机融合点云、关键点、规划整臂留影与真实 TCP 轨迹。橙色是真实反馈，蓝色是规划预览。相机图像下方给出采集时间、时效与坐标系。停止按钮请求轨迹取消与驱动停机；失能是独立操作。

当前保留已删除点云规则抓取的状态，工作台已接入许可版 AnyGrasp。使用当前会话两相机已确认的冻结 SAM 实例点云，目标 2 mm、背景 3 mm 体素，背景最多 30,000 点；掩膜与快照哈希、时间戳绑定，不读取 RViz 拖动偏移，也不使用距 RS1 表面 8 mm 补点作为抓取输入。

AnyGrasp 入口 `infer_directional` 分别查询上方（向下法向 15°）和侧方（每 30° 一个方向，最终接近和闭合均离桌面不超过 10°）候选，再以 Piper URDF 掌部、双指网格检查接近和闭合过程的桌面净空至少 3 mm。桌面来自当前场景模型。全部原始模型候选和拒绝原因保存在会话下 `anygrasp-*/audit.json`。

方向与桌面通过后检查预抓取和抓取端点 IK、端点自碰撞，再进入原有 ReKep 路径生成、稠密 IK 和全路径检查；一个候选路径失败可尝试下一个模型候选。侧向抓取的接近约束沿候选接近轴，不能套用垂直下降约束。接触证据独立保留，未证实的候选只允许 `EXPERIMENTAL_PREVIEW_ONLY` 计算草案，不能获得执行权限。若初始夹爪开合检查失败，可留下诊断路径草案，但不会安装为通过检查的预览。`proposed_joint_path.npy` 是检查前草案，不代表有效轨迹。

双相机审阅仍保留点云和上下拖动轴，实体青色／紫红色方块已去除；不改变点云数据。

使用顺序：连接 CAN → 使能 → **1 启动相机**，查看 RS1、RS3 与 RViz 点云 → **2 生成关键点与固定约束**：SAM＋DINOv2 产生编号点，VLM 只选择蓝色方块和黄色圆盘上的两个编号；本地把对应实测三维坐标填入固定四阶段约束 → **3 逐段执行约束**：预览并确认预抓取、抓取、抬升并运输、下放释放。RS1 可切换编号快照，RViz 同时显示静态编号点和 DINOv2 更新的关键点轨迹；轨迹仅供观察，不代替执行前场景复核。抓取段完成后，操作者须点击“已夹稳”才能预览运输；未夹稳返回抓取段。每段从真实关节反馈重新规划，预览阶段不发送运动或夹爪命令，执行时复核同一条已显示轨迹。


使能前先核对控制器状态和实际指示灯。`teach_status=1` 表示控制器报告示教中；仅在绿灯常亮且机械臂已托稳时，按设备手册单击绿键退出，勿双击回放示教轨迹。若绿灯不亮而状态仍为 `teach_status=1`，先核对控制器停机状态及设备故障；该协议状态不代表机身一定有实体急停按钮，不要仅凭状态字段操作绿键。`arm_status=1` 时工作台拒绝使能；停机恢复可能导致机械臂失去支撑，须由现场人员确认支撑和工作区后处理。使能后若实测关节角超出 URDF 模型限位，工作台仍禁止规划和运动；使能本身不再因这个模型检查提前失败。

启动相机后若无画面，先查看窗口底部的相机状态和图像时间。相机图像应发布在 `/rs1/color/image_raw` 与 `/rs3/color/image_raw`；若曾运行修复前的工作台，它们可能错误地发布在 `/rekpiper/supervised/rs1/...` 与 `/rekpiper/supervised/rs3/...`。在启动工作台的终端按 `Ctrl+C`，等待子相机进程退出，再重新运行上述 `roslaunch` 命令，并重新点击连接、使能和启动相机。重启会撤销旧会话的预览，不会自动下发运动。

相机画面可独立查看；手工指定的实验外参文件已加载、文件内容与相机序列号保持一致时，可先生成仅供检查的轨迹预览。使用相机关键点三维坐标驱动机械臂前，仍需展开“执行前坐标复核”，冻结双相机图像并录入至少四个覆盖两个高度及抓放区域的独立基座已知点。未通过复核时“执行本段”不可用，预览不授予运动权限。

实验外参文件、相机序列号、`base_link_T_arm_base` 安装关系与速度位于 `src/rekpiper_bringup/config/supervised_workbench.yaml`。当前 `base_link_T_arm_base` 是需要已知点实测复核的安装假设。复核只在本会话内生效；不改变生产外参状态。会话原始数据及事件、规划、人工确认、关节反馈保存在 `runtime/supervised/<会话 ID>/`。接口在 `/rekpiper/supervised/`，执行 Action 仅接受会话、阶段和预览 ID。

软件验证：`catkin_make`、`catkin_make run_tests`、`catkin_test_results build/test_results`、`python3 tools/check_repository.py --source-root "$PWD"`、`python3 tools/check_launch_paths.py --package-root "$PWD/src" --dump-shadow`。无设备界面截图可由 `/usr/bin/python3 src/rekpiper_execution/test/render_supervised_panel_fixture.py` 生成，输出 `runtime/evidence/supervised_workbench_fixture.png`。这张截图回放仓库已有的 RS1/RS3 图像并使用 RViz 占位框；它只验证窗口布局。真机连接、限位/碰撞监测与四段抓放需在现场另行验收。

## 路径搜索与最终验收

工作台前两阶段使用已选 AnyGrasp 候选的预抓取／抓取 TCP 位姿，后两阶段使用任务模板算出的目标；这些固定目标仍须满足本阶段约束，省去重复的 SubgoalSolver 全局搜索。路径仍使用 `official_exact` 评价：优化时按官方 ReKep 的方式在控制点计算 IK；控制点间隔为 0.10 m／15°。搜索结束后再对完整插值路径执行密集 IK、关节跳变及碰撞检查。优化分数为零或端点 IK 成功均不代表路径验收通过。

一个候选的规划或验收失败时会尝试下一个模型候选；取消操作立即停止。工作台总规划预算为 600 秒，超时不产生可执行路径。
