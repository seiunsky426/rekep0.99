# 蓝色点云三组 AnyGrasp 对比

独立实验入口使用固定的一对 RGB-D 帧，颜色筛选蓝方块，比较 RS1、RS3 和融合输入。三组都保留各自的场景背景；不调用 SAM/VLM，不生成规则抓取，不调用机械臂控制、碰撞检查或 IK。

## 采集与推理

使用工作台配置中的手工外参及相机序列号；相机需要已经发布新鲜 RGB、对齐深度和内参。采集窗口为 30 秒，最大时间跨度 25 ms。输出目录必须是新的。

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
comparison_dir="runtime/evidence/blue_cloud_anygrasp_compare_$(date +%Y%m%dT%H%M%S)"
/usr/bin/python3 tools/compare_blue_cloud_anygrasp.py capture --output "$comparison_dir"
/usr/bin/python3 tools/compare_blue_cloud_anygrasp.py prepare --output "$comparison_dir"
OMP_NUM_THREADS=8 runtime/bin/python tools/compare_blue_cloud_anygrasp.py infer --output "$comparison_dir"
/usr/bin/python3 tools/show_blue_cloud_anygrasp.py --output "$comparison_dir" --record
```

捕获文件保存原始图像、深度、K、时间戳、外参副本和 SHA256。所有模型输入的主坐标系均为 `arm_base`，使用真实矩阵变换；工作空间范围按其原配置的 `base_link` 坐标解释。推理入口统一采用 RS1 光学坐标轴，输出保存原生 AnyGrasp 抓取姿态到 `arm_base` 的变换，不附加 Piper TCP 深度偏移。

颜色阈值为 OpenCV HSV：H 78–115、S>90、V>45；使用工作空间内最大蓝色连通区域、3×3 腐蚀和 20 邻域均值距离的 2.5 标准差离群点剔除。原始和清理后的有效目标点均须至少 120；不足时该组记录错误，融合组也不会假装双相机输入有效。腐蚀边缘及被剔除的目标点不会重新归为背景。

目标/背景体素分别为 2/3 mm，融合保存来源位标记（1=RS1、2=RS3、3=两者）；背景排除与目标重合的体素。各组背景最多 30,000 点，随机种子为 0。模型使用最大夹宽 70 mm、非稠密生成、无接近方向限制及关闭碰撞检测；适配器 `max_candidates=None` 保留全部 NMS 后候选，其他调用的默认上限仍是 50。

每组仅运行一次。记录的是实测单次耗时，第一组可能包含 CUDA 首次运行开销，不据此得出模型速度排名。固定可控随机种子不保证 CUDA 原生扩展逐位确定性。

## RViz 与回放

`--record` 打开专用 RViz 窗口，按相同视角分别保存三组 top1/top5，共六张真实窗口截图和六个 bag。目标为蓝色、背景为淡灰，彩色夹爪轮廓及箭头分别代表候选与接近方向，颜色对应旁边排名。夹爪轮廓遵循 AnyGrasp 原生 x 接近轴、y 闭合轴，手指前端位于 x=depth。

截图完成后默认停留在融合组前五候选，可自由缩放。记录过程中，每张图的相机参数会自动重置并核验。可在另一个终端核验所有输入、姿态、截图及回放文件：

```bash
/usr/bin/python3 tools/compare_blue_cloud_anygrasp.py verify --output "$comparison_dir"
```

可切换显示：

```bash
rosparam set /blue_cloud_anygrasp_comparison_display/case rs1
rosparam set /blue_cloud_anygrasp_comparison_display/top_k 1
```

`case` 取 `rs1`、`rs3`、`fused`；`top_k` 取 1 或 5。实时发布器运行时不要同时向相同话题播放 bag。

发布器退出后，可在两个终端回放一组记录：

```bash
DISABLE_ROS1_EOL_WARNINGS=1 rviz -d "$comparison_dir/comparison.rviz" \
  /tf:=/rekpiper/blue_cloud_anygrasp_compare/tf \
  /tf_static:=/rekpiper/blue_cloud_anygrasp_compare/tf_static
rosbag play --keep-alive "$comparison_dir/rs1/top1.bag"
```

仅在私有显示话题上提供 `comparison_reference → arm_base` 的恒等变换，避免 RViz 在没有全局 TF 时报告缺失坐标系；不发布到正式 `/tf` 或 `/tf_static`，不修改相机标定。

`summary.json` 给出点数、候选数、最高分和耗时；`rs1/`、`rs3/`、`fused/` 各保存输入 NPZ、PLY、姿态 JSON、截图及 bag。零候选时显示 `NO CANDIDATES`，不伪造姿态。候选数和网络分数都不是抓取成功率，也不代表机械臂可达或路径可执行。

## 已冻结点云的上方/侧方检查

在原三组 `input.npz` 上运行，不重新采集、不改变点云。以下入口新建六组结果目录：

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 runtime/bin/python tools/compare_directional_anygrasp.py \
  --source "$comparison_dir" --output "$directional_dir"
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 tools/show_blue_cloud_anygrasp.py \
  --output "$directional_dir" --cases rs1_top rs1_side rs3_top rs3_side fused_top fused_side \
  --case fused_top --top-k 1 --record
```

`directional_dir` 为新的实验目录。每组 top/side 接近方向都从同一个冻结 RS1 背景的局部桌面法向定义；RS1 是此工作台原有桌面参考。拟合失败直接停止，不假设桌面是 z=0，也不替换为其他相机平面。平面及拟合 RMS 保存在 `table.json`，这一共享平面不代表 RS3 的外参已经独立验收。

上方约束为向下法向 15° 锥；侧方沿桌面每 30° 提交一个方向，共 12 个 15° 锥，然后要求接近轴和闭合轴都在桌面方向 10° 内。多个调用的候选再统一 NMS，避免重复计数。仍关闭 SDK 的近似夹爪碰撞筛选，显式使用 Piper URDF 的掌部和两根手指网格进行桌面检查。

`geometry_pass` 只表示方向、实际夹宽/插入深度及桌面净空检查通过。通过真实 TCP 轴映射，检查 60 mm 预抓取距离的预张开接近过程和闭合过程，所有网格须离平面至少 3 mm。固定朝向的直线接近与棱柱关节开合中，网格顶点到平面的距离为仿射函数，所以端点顶点的最小值能覆盖整段过程，不只检查抓取中心。

`contact_supported` 是独立的观测证据检查：两个预期接触点都要在目标表面 5 mm 内、可靠局部法向与闭合轴偏角不超过 20°，且目标点位于两指之间。没有观测到背面时可能无法满足，因此该字段失败不等同于已证明物理上无法抓取。

RViz 显示的是预张开状态的真实 URDF 碰撞网格，不再使用原生 SDK 的简化 U 形夹爪；绿色矩形为实测桌面平面边界提示。界面和文件分别标记几何通过数、接触支持数。零几何通过时显示 `NO GEOMETRY PASS`，不会用淘汰候选充数。原始候选和逐项拒绝原因保存在每组 `audit.json`。

不做整臂 IK、完整场景碰撞或真实抓取；几何通过不授予运动权限。显示切换仍用 `/blue_cloud_anygrasp_comparison_display/case`，值改为上述六个组名。
