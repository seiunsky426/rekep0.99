# M0.2–M0.5 软件部署与验收记录

验收日期：2026-09-10；文档归档：2026-09-11。工程：`/home/zhengsihze/Rekep_v1.0-main`。

**M0.1–M0.5 全部通过，M0 为 5/5 完成（软件环境与离线验收）。** 本报告接续 [M0.1 环境记录](M0_1_ENVIRONMENT.md)，目标状态见 [部署目标清单](REKEP_DEPLOYMENT_WORKFLOW_BASELINE.md)。

本报告只验收环境、模型计算、源码一致性、设备命名及构建接口。真实相机数据、标定、机器人反馈、在线地图和物理执行仍按 M1–M11 逐项验收。没有启动相机、Piper/CAN 节点或发送运动指令。

## M0.2：模型与扩展

使用当前 `runtime/bin/python`，Python 3.8.10、PyTorch 2.4.1+cu121、CUDA Toolkit 12.1。三份 Python 锁去重后共 **158 项**，保留此前全部依赖版本；平台锁只新增 Cython 0.29.37，保证能在安装模型依赖前准备 grasp-nms 的源码包。

| 组件 | 固定来源/版本 | 离线 CUDA 检查 |
|---|---|---|
| SAM ViT-H | segment-anything `6fdee8f2727f4506cfbbe553e23b895e27956588` | FP16 自动掩码计算，128×128 输入得到有效 bool 掩码 |
| DINOv2 ViT-S/14 reg4 | `e1277af2ba9496fbadf7aec6eba56e8d882d1e35` | 输出 patch features 为 1×16×384，全部有限 |
| Cutie v1.0 | `7db8f9c9133f4884a9a13d865f9e3df20868d569` | 种子帧和下一帧推理，128×128 标签含 0/1 |
| cuRobo v0.7.6 | `2fbffc35225398cf9d5f382804faa9de2608753b` | 使用本工程 Piper URDF、关节值和合成深度计算 robot mask |
| nvblox_torch | wrapper `8c73aa113b8aa6cc897f8fec029f895a481e9676`；core `7bda93e993594ca69c479dde64213ac35835eb84` | 三帧深度融合、ESDF 更新和距离查询 |
| AnyGrasp | 本机 licensed SDK，二进制/检查点 SHA256 纳入本地清单 | 许可 PASSED；MinkowskiEngine 稀疏卷积、pointnet2 采样、抓取推理通过；50 个候选 |

SAM 的 source-root 改为 `REKPIPER_VENDOR_ROOT/segment-anything`，保留来源与 commit 检查。此机 8 GB 显卡上 FP32 ViT-H 推理超出可用显存；适配器在加载原检查点后转换 CUDA 参数为 FP16，并使用 autocast。ROS 配置为 `scene/sam_use_float16: true`，CPU 仍使用 FP32。小样本采用降低阈值来确认计算链能运行，正式阈值没有降低；分割质量和多节点同时运行的资源预算留待 M4/M10 验证。

`runtime/vendor` 已由指向旧工作区的整体符号链接改为当前工程拥有的本地源码/SDK 副本，SAM 目录统一为 `segment-anything`。`rekpiper_vendors.pth` 只加入指定源码目录及本地扩展路径，没有引入旧 Python 的 site-packages。原有 GraspNetAPI 完整发行版的 egg-info 已存入本轮证据目录；当前使用仓库裁剪后的推理源码，不注册包含训练/评估依赖的旧发行版元数据。

nvblox_torch 在本机用当前 PyTorch、CMake 3.27.9、CUDA 12.1 和 sm_89 重新编译。nvblox core 及其本地依赖从既有产物复制到 `runtime/native`，保留源码和二进制哈希；CMake 导出路径已重定位。MinkowskiEngine 0.5.4 和 pointnet2 的既有扩展复制到当前 vendor 后，在当前解释器中完成真实 CUDA 运算验证。不能将这些二进制视为任意 Python/CUDA 版本均可复用。

cuRobo 的 CUDA 扩展缓存在 `runtime/cache/torch_extensions`。Piper 配置补齐空的 `self_collision_buffer` 和 `self_collision_ignore`，解决初始化时对 None 调用 copy 的错误；空忽略表保留所有连杆对。`bootstrap_anygrasp_runtime.sh` 已改为使用正式 runtime，在本地源码树中构建扩展，不再另建带系统 site-packages 的环境。它是重建入口，本轮使用的 MinkowskiEngine/pointnet2 产物以实际加载和运算检查为证据。

本地 `runtime/site-config/M0_ASSETS.lock.json` 保存四份权重、源码、SDK 和原生扩展等 1724 个文件的 SHA256，逐项检查通过。它是软件资产基线，**不是现场签名批准**；`third_party/VENDOR.lock.yaml` 的 `approved: false`、`SITE_REQUIRED` 以及真实执行的签名检查继续保留。

## M0.3：官方源码一致性

正式加载目录仍为 `third_party/ReKep`，期望提交保持 `63c43fdba60354980258beaeb8a7d48e088e1e3e`。恢复前已对 11 个不匹配文件逐个验证：本地参考副本的字节 SHA256 与原锁一致，去掉文档字符串后的 AST 与中文注释版一致。

没有修改上游期望哈希，也没有关闭来源检查。中文注释版本保存在：

- [完整注释文件归档](reference/ReKep_chinese_annotations_20260910.tar.gz)
- [相对锁定原版的注释补丁](reference/ReKep_chinese_annotations_20260910.patch)

归档用于独立阅读副本；将补丁重新应用到正式上游目录会再次触发完整性检查。仓库审计及实际 `load_official_core()` 加载器均通过。

## M0.4：rs1/rs3 生产接口

正式相机对为 rs1=`346522071783`、rs3=`934222070377`。迁移覆盖 launch、相机投影/融合、DINO/Cutie 跟踪、对象状态、地图监督、标定采集、release schema/设备绑定及相关测试。

`SafeMappingStatus.rs2_valid_frames` 已改为 `rs3_valid_frames`，所有依赖包已重建。旧消息录包/外部订阅者需要按新消息契约转换或重新编译；不会把旧 rs2 数据自动当成 rs3。

相机启动和 TF 发布共用同一组 serial 参数。已用替代 serial 展开 launch，验证 rs3 驱动和外参校验器收到完全相同的覆盖值。测试覆盖 rs3 拒绝旧设备序列号的外参。

rs2 外参文件保持原字节，生产 launch 不再引用；原 rs1/rs2 工作区测量另存 `config/archive/recognition_workspace_rs1_rs2_20260728.yaml`。当前工作区保留原边界供离线检查，状态为 UNVERIFIED，不把原 rs2 测量改名为 rs3 测量。

rs3 正式外参是绑定自己 serial/frame 的 **UNCALIBRATED 占位配置**，未发布 TF。已有独立 launch、内参和手眼结果原样复制至 `runtime/site-config/camera_provenance`，保存源路径与 SHA256。640×360 内参与当前流配置、`arm_base` 与 `base_link` 的关系仍需 M2 验证。

## M0.5：构建与接口

- 当前正式 Python/nosetests 构建全部 13 个 Catkin 包。
- `rekpiper_msgs.msg`、`rekpiper_msgs.srv`、`piper_msgs.msg` 来自当前 `devel`。
- 仓库审计 `clean=true`，launch 检查 `valid=true`。
- 38 个 launch 节点可执行文件引用全部可解析。
- 单元测试包括 SAM 来源保护、rs3 身份拒绝和原有安全测试；最终 **209 tests，0 errors，0 failures，0 skipped**。

## 复查命令

```bash
cd /home/zhengsihze/Rekep_v1.0-main
source setup.bash
python tools/check_m0_environment.py
python tools/check_m0_assets.py
for model in sam dinov2 cutie curobo nvblox anygrasp; do
  python tools/check_m0_models.py "$model" || break
done
python tools/check_repository.py --source-root "$PWD"
python tools/check_launch_paths.py --package-root "$PWD/src" --dump-shadow
catkin_make -DPYTHON_EXECUTABLE="$REKPIPER_RUNTIME_ROOT/bin/python" \
  -DNOSETESTS="$REKPIPER_RUNTIME_ROOT/bin/nosetests" -j2
source setup.bash
catkin_make run_tests -j2
catkin_test_results build/test_results
```

GPU 小样本按独立进程顺序执行。它们不会启动 ROS 节点，也不代表实际感知、双相机帧率、在线闭环或机械臂操作已验收。

本机重新安装 Python 依赖时，先装核心锁与平台锁，再装模型锁：

```bash
python -m pip install --no-index --find-links runtime/wheelhouse \
  --require-hashes -r requirements.lock
python -m pip install --no-index --find-links runtime/wheelhouse \
  --find-links runtime/platform-wheelhouse --find-links runtime/model-wheelhouse \
  --require-hashes -r requirements.platform.lock
python -m pip install --no-index --find-links runtime/wheelhouse \
  --find-links runtime/platform-wheelhouse --find-links runtime/model-wheelhouse \
  --no-build-isolation --require-hashes -r requirements.models.lock
```

模型锁由 pip-tools 7.4.1 从 `requirements.models.in` 生成，使用核心/平台锁约束。`--no-build-isolation` 使用已锁定的 NumPy、Cython、setuptools 和 wheel 准备 grasp-nms；没有关闭安装包 SHA256 检查。模型安装目录中的 54 个原始安装包已逐个对照官方 PyPI 发布的 SHA256，失败的截断下载未用于安装。

本地源码入口、native 目录、权重、SDK 和许可仍需按本机资产清单保留；这些内容不进入 Git。换机器或 ABI 后应重新构建扩展并重新验证授权，不复制现场批准结论。

## 证据

本轮目录：[runtime/evidence/m0_2_5_20260910T140405Z](../runtime/evidence/m0_2_5_20260910T140405Z)。保存模型 JSON/日志、源码恢复前后与 AST 比较、依赖来源校验、nvblox 构建/动态库路径、rs3 参数传递、Catkin 完整结果以及变更前副本。

- [环境最终检查](../runtime/evidence/m0_2_5_20260910T140405Z/environment_final.json)、[资产哈希检查](../runtime/evidence/m0_2_5_20260910T140405Z/assets_final.json)
- [六个模型的最终 CUDA 记录](../runtime/evidence/m0_2_5_20260910T140405Z/final_models)
- [上游恢复与 AST 对照](../runtime/evidence/m0_2_5_20260910T140405Z/upstream_restoration.json)、[仓库审计](../runtime/evidence/m0_2_5_20260910T140405Z/repository_final.json)
- [rs3 序列号传递](../runtime/evidence/m0_2_5_20260910T140405Z/rs3_serial_forwarding.json)、[生成接口来源](../runtime/evidence/m0_2_5_20260910T140405Z/generated_interfaces_final.json)
- [launch 检查](../runtime/evidence/m0_2_5_20260910T140405Z/launch_final.json)、[Catkin 完整测试结果](../runtime/evidence/m0_2_5_20260910T140405Z/catkin_results_final.txt)
