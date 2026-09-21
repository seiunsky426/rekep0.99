# M0.1：当前工程的独立运行环境

工程根目录：`/home/zhengsihze/Rekep_v1.0-main`。本文件管理基础环境；模型/扩展、上游源码、rs3 迁移和完整系统验收分别属于 M0.2–M0.5 及后续模块。

**当前 M0 已完成 5/5。** 以下使用说明已同步三份依赖锁与本地 vendor；末尾保留 M0.1 单独完成时的历史验收。当前 158 项锁定依赖、六项模型检查和 209/209 测试结果见 [M0.2–M0.5 最终验收记录](M0_2_TO_M0_5_COMPLETION.md)。

## 日常入口

在新终端中执行：

```bash
cd /home/zhengsihze/Rekep_v1.0-main
source setup.bash
python tools/check_m0_environment.py
```

检查工具会核对 Python 3.8、ROS Noetic、目录、三份依赖锁、核心模块的实际导入路径、13 个 ROS 包的来源、编译工具、CUDA 小矩阵和 pip 依赖关系。工具不启动 ROS 节点、不访问相机、不发送 CAN 指令。

## 目录与入口合同

| 内容 | 当前配置 |
|---|---|
| Python 环境 | `<工程>/runtime`；include-system-site-packages=false |
| Python 解释器 | `<工程>/runtime/bin/python` |
| 模型目录 | `<工程>/runtime/models`，链接至 `/home/zhengsihze/rekpiper_runtime/assets/models` |
| vendor 目录 | `<工程>/runtime/vendor`，当前工程拥有的源码/SDK 副本 |
| 现场配置 | `<工程>/runtime/site-config` |
| 信任材料目录 | `<工程>/runtime/trust`；目录存在不表示已有有效批准材料 |
| 数据与 ROS 日志 | `<工程>/runtime/data`、其下 `ros` |
| Python/CUDA 缓存 | `<工程>/runtime/cache` |
| 本地安装包 | `<工程>/runtime/wheelhouse`、`runtime/platform-wheelhouse`、`runtime/model-wheelhouse` |
| 本轮记录 | `<工程>/runtime/evidence/m0_1_20260910T133717Z` |
| 官方 ReKep | `<工程>/third_party/ReKep`，M0.3 已恢复至原锁定字节 |

模型链接复用既有权重，vendor 使用本地副本；均不导入旧环境的 site-packages。旧 `/home/zhengsihze/rekpiper_runtime/env/rekpiper-py38` 保留。

`setup.bash` 激活独立 venv，设置 PYTHONNOUSERSITE=1，加载可选的 `runtime/site-config/environment.sh`，再加载 ROS/当前 devel，并保证当前 src 位于 ROS_PACKAGE_PATH。现场环境文件负责 CUDA_HOME=/usr/local/cuda-12.1 及缓存路径；这是本机文件，受 `.gitignore` 排除。

不同工程之间切换时使用新终端；脚本保留显式 REKPIPER_* 覆盖值。检查工具会拒绝错误的工程根目录、解释器或核心模块来源。

## 依赖锁

- `requirements.in` / `requirements.lock`：原有 10 项核心直接依赖及其完整依赖树。
- `requirements.platform.in` / `requirements.platform.lock`：PyTorch 2.4.1、torchvision 0.19.1、ROS Python 辅助包、Catkin 测试工具、Ninja、Cython 和已有 Piper SDK 接口。以核心锁为约束，不调整核心版本。
- `requirements.models.in` / `requirements.models.lock`：模型推理及原生扩展的 Python 依赖，以核心锁和平台锁为约束；源码、权重、CUDA 扩展另按本机资产清单准备。
- ROS 本体和 CUDA Toolkit 使用本机已安装的 ROS Noetic 与 CUDA 12.1；不能用 pip 安装成功代替二者检查。

补充锁由 Python 3.8、pip 24.0、pip-tools 7.4.1 生成，不手写 wheel 哈希。独立的锁生成工具环境位于 `runtime/lock-tools`，不进入正式 Python 的导入路径。

## 本机离线重建命令

以下命令针对已有本地安装包目录；不要删除或覆盖现有环境来试跑。需要新建副本时将 venv 路径换成独立目录，并保留原环境。

```bash
python3.8 -m venv runtime
source runtime/bin/activate
python -m pip install --no-index --find-links runtime/wheelhouse pip==24.0
python -m pip install --no-index --find-links runtime/wheelhouse \
  --require-hashes -r requirements.lock
python -m pip install --no-index \
  --find-links runtime/wheelhouse --find-links runtime/platform-wheelhouse \
  --find-links runtime/model-wheelhouse \
  --require-hashes -r requirements.platform.lock
python -m pip install --no-index \
  --find-links runtime/wheelhouse --find-links runtime/platform-wheelhouse \
  --find-links runtime/model-wheelhouse --no-build-isolation \
  --require-hashes -r requirements.models.lock
source setup.bash
python -m pip check
python tools/check_m0_environment.py
```

上述离线命令验证 Python 依赖；模型链接、vendor 副本、原生扩展及现场 environment.sh 需要按本机目录合同准备。另一台电脑不得直接复制本机外参或审批记录。

本次正常 pip HTTPS 连接出现 TLS EOF，而系统 HTTPS 读取可用，因此安装时使用仅绑定 127.0.0.1 的临时中转：中转以验证证书的 HTTPS 访问官方 PyPI，核心包继续强制核对原锁文件 SHA256，随后从本地目录安装。中转会在命令结束后关闭，不是运行系统的依赖。

## 使用正式 Python 构建

```bash
source setup.bash
catkin_make -DPYTHON_EXECUTABLE="$REKPIPER_RUNTIME_ROOT/bin/python" \
  -DNOSETESTS="$REKPIPER_RUNTIME_ROOT/bin/nosetests" -j2
source setup.bash
catkin_make run_tests -j2
catkin_test_results build/test_results
```

显式指定 NOSETESTS，避免 Catkin 选中系统 `/usr/bin/nosetests3` 后回到另一套 Python 依赖。M0.3 已恢复上游源码，M0.5 的完整测试通过，相关校验保持启用。

## M0.1 单独完成时的历史验收

**2026-09-10 验收通过：M0.1 已完成。** 最终在清除旧工程变量的新 shell 中加载 setup.bash，检查结果为 m0_1_complete=true、errors=[]。

| 验收项 | 结果 |
|---|---|
| 独立 Python | runtime/bin/python，Python 3.8.10；禁用系统/用户 site-packages |
| 核心直接依赖 | 10/10 与 requirements.in 精确一致 |
| 完整版本锁 | 核心锁 88 项、补充锁 46 项，去重后 123 项全部匹配 |
| 核心模块来源 | 实际导入均来自当前 runtime |
| ROS 与目录 | ROS Noetic；13 个包均解析至当前源码；所需根目录存在 |
| 工具链 | CUDA Toolkit 12.1；Ninja 来自当前 runtime |
| GPU | RTX 4070 Laptop GPU；torch 2.4.1+cu121；矩阵计算结果 4096.0 |
| 依赖关系 | pip check：No broken requirements found |
| Catkin | 使用当前 runtime 的 Python 和 nosetests，13 包构建及消息导入通过 |
| launch | 路径与 shadow 参数展开检查通过，未运行节点 |
| 现有测试 | 207 项中 203 通过，2 errors、2 failures；与首轮核对相同，均由 M0.3 上游哈希问题触发 |

在 M0.1 单独完成时，原 requirements.in、requirements.lock 及上游 Python 源码与该轮开始时的 SHA256 一致，M0.2–M0.5 尚未完成。后续上游恢复及完整通过的结果见文首最终验收链接。现场批准状态保持不变。

证据目录：[`runtime/evidence/m0_1_20260910T133717Z`](../runtime/evidence/m0_1_20260910T133717Z)。主要文件：

- [最终环境检查](../runtime/evidence/m0_1_20260910T133717Z/environment_check_final.json)
- [核心 10 项版本对照](../runtime/evidence/m0_1_20260910T133717Z/core_versions_final.json)
- [环境配置与锁文件哈希](../runtime/evidence/m0_1_20260910T133717Z/environment_sources.sha256.json)
- [完整 Python 版本清单](../runtime/evidence/m0_1_20260910T133717Z/python_freeze.txt)
- [构建日志](../runtime/evidence/m0_1_20260910T133717Z/catkin_build.log)
- [测试结果](../runtime/evidence/m0_1_20260910T133717Z/catkin_results.txt)

当前补充锁的生成命令如下；安装包目录中的文件已完成来源及哈希核对：

```bash
source setup.bash
runtime/lock-tools/bin/pip-compile --cache-dir runtime/cache/pip-tools \
  --allow-unsafe --generate-hashes --strip-extras --resolver=backtracking \
  --no-index --find-links runtime/wheelhouse --find-links runtime/platform-wheelhouse \
  --find-links runtime/model-wheelhouse --pip-args=--no-build-isolation \
  --no-emit-find-links --no-emit-index-url --no-emit-trusted-host \
  --output-file requirements.platform.lock requirements.platform.in
```
