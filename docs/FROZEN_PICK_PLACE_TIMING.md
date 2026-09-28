# 固定场景的抓取、运输、放置规划计时

本流程沿用 `Eye_To_Hand_ws/src/rekep_moveit_preview`，只做规划和预览。
桌面、蓝色方块、黄色圆盘在采集后保持静止。采集一次双 RealSense
RGB-D 和被动 CAN 关节反馈，固定融合点云、物体关键点、模型和原始时间戳。
运行结果不代表实体已经抓住物体；运输和放置使用明确标记的假设附着关系。

## 固定输入

采集完成后使用 `preview.py freeze --session <采集目录> --config <配置文件>`。
`frozen_scene.json` 记录输入文件 SHA256。每次新运行都会核验并复制这些文件；
不会重新采集、更新时间戳或覆盖原场景。文件缺失、空点云和哈希变化立即报错。

关键点沿用昨天的测量方法：RS1 蓝色物体中心、黄色圆盘中心和蓝色包围盒角点。
双相机融合点云用于 AnyGrasp 输入；这不等价于重新运行 SAM/DINO/VLM。
如果 `--frozen-scene` 指向已有 AnyGrasp 结果的同场景运行目录，还会绑定哈希后复用抓取候选，跳过重复模型加载。
桌面模型、工作区边界及标定文件另有哈希绑定，变化时立即拒绝。其原有验收状态不改变。

## 运行

先加载 ROS 和两个工作区的 Python 环境，使用独立端口运行仅预览的 MoveIt：

```bash
source /opt/ros/noetic/setup.bash
source /home/zhengsihze/Eye_To_Hand_ws/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11334
export ROS_HOSTNAME=127.0.0.1
roslaunch /home/zhengsihze/Eye_To_Hand_ws/src/rekep_moveit_preview/launch/moveit_preview.launch \
  session:=<固定场景绝对路径> --port=11334
```

另一个终端使用相同 ROS 环境，运行：

```bash
/home/zhengsihze/Rekep_v1.0-main/runtime/bin/python \
  /home/zhengsihze/Eye_To_Hand_ws/src/rekep_moveit_preview/scripts/preview.py plan \
  --frozen-scene <固定场景绝对路径> --session <全新运行目录> --config <配置文件>
python3 tools/summarize_frozen_planning.py <运行目录>
```

配置 `grasp_only: false` 请求三个阶段；`true` 只请求抓取。
抓取位姿来自本地 licensed AnyGrasp；抓取路径调用官方 PathSolver；
运输和放置另外调用官方 SubgoalSolver；若返回目标未通过原约束，先在该目标附近进行局部约束修正，并用实际 IK/FK 位姿再次核验相同约束。修正失败才尝试其他种子。每阶段经过 MoveIt 连续关节转换、
全臂/持物碰撞、FK 与任务约束检查、时间参数化及速度/加速度检查。

## 性能与失败恢复

- `STAGE wait` 和 `STAGE compute` 分开显示等待和工作阶段的单调时钟墙钟耗时。
  后者包含进程调度，不声称等于纯 CPU 时间。子进程的细分计时不能与父阶段相加。
- 保存 AnyGrasp 模型加载与推理、路径全局/局部优化、关节转换、碰撞审计、重新定时的计时。
- 服务 URI 消失或改变时报告 `service_unavailable` / `service_restarted`。
  求解期间每次监督轮询检查服务；中断子进程后丢弃结果。
- 点云缺失直接拒绝。AnyGrasp 进程异常退出直接返回失败；120 秒仅是仍存活计算进程的上限。
- 速度/加速度超限先统一拉长同一条关节路径的时间。原生时间数据不合格时，
  最多以 1、0.5、0.25 的缩放重新定时；耗尽后停止，不重跑空间路径。
- 全局候选已通过官方求解器检查、但随后可选局部优化失败时，保留原全局候选继续做完整关节和碰撞检查；不会因为局部优化失败直接丢弃它。
- 固定抓取终点先做夹爪几何/桌面间隙检查，不能通过的候选直接排除，避免三次重复路径搜索。
- 仅完整审计通过的路径发布热启动缓存。缓存绑定场景栅格、起始关节、目标、
  关键点、碰撞样本、模型、约束、配置及求解器源码。命中时先局部优化并重新执行所有审计。
  局部失败才回退全局搜索；关节路径失败可换种子或候选。
- 时间参数化前已审计的关节几何若完全相同（仅去除连续重复点），复用几何审计；
  任何实际位置变化都重新做几何检查。速度和加速度仍独立检查。

预览后可用原有 `replay_node.py --session <运行目录> --no-controls` 发布
冻结点云、路径和全机械臂回放数据；没有界面也不会接入硬件控制。
