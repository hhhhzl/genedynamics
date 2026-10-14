# P3 官方源码与 solver 接入

更新：2026-10-14。用户确定：源码放仓库根 `baselines/`，实现进入现有
`genedynamics/solvers/`；先原生 PyTorch，再做数值对齐，JAX 后续按需要决定。
本文记录源码接入，论文 draft 不承载这些开发记录。
本轮完整训练/适应接口、原生资源和真实四任务执行已继续实现，最新逐路状态见
[P3 执行记录](cross-embodiment-p3-execution.md) 与运行生成的 P3 审计 JSON。
下文 40/40 对齐数据保留为最初源码核心版本的历史证据。

## 1. 当前交付范围

| 方法 | 已移植到 solver 的核心 | 尚未完成的原生链条/资格 |
| --- | --- | --- |
| FADA | 原生 Planner–IDM/LoRA；G1 发布 oracle、真实 PPO/严格 shadow DAgger、目标记录/LoRA 与完整续训；项目任务训练链 | mjwarp 属模拟器改编；学生稳定性与充分训练待验，任务公共物理 chart 修正；原生 IsaacSim 定量未复现 |
| Human2Sim2Robot | actor/RNN/normalizer；原生 PpoAgent、轨迹 reward 与公开名义目标模型 RL；T1/T3 真实训练/完整续训/目标执行 | 人视频/孪生资产和发布模型仍 assets_pending；held-object/shared-controller 改编未替代原生人手探索/FABRICS |
| ACP | 原生真实 CLIP+force-ViT/发布权重/示范标签、Mod1/EMA/loss/续训；真实各向异性控制与 T3 state-history learner | 硬件成功率未复现；任务 source sampling 和示范/控制族兼容仍需资格，不以 loss 下降替代任务表现 |
| Play2Perfect | 固定官方源码；两个发布checkpoint严格加载；原生Isaac tight-insertion seed42/8环境完整release/retract闭环8/8成功 | 通过发布checkpoint原生功能资格；未复现完整pretraining/fine-tuning或论文aggregate score，未删除in-hand核心后塞入固定夹持主表 |

数值验证使用小规模模型、构造的输入和本次生成的权重，直接与固定上游模块
比较；没有把它写成作者发布 checkpoint 验证或论文成绩。完整 P3 尚未完成，
项目四任务的适配资格也未由这些测试放行；新增物理/训练资格独立记录于上述执行文档。

## 2. 文件分工

```text
baselines/{FADA,Human2Sim2Robot,ACP,Play2Perfect}/  官方参考源码（git ignored）
requirements/baselines/sources.lock.json          URL、commit、许可证 SHA、检出范围
genedynamics/solvers/single/fada/                 模型、推理、checkpoint 与来源映射
genedynamics/solvers/single/human2sim2robot/       actor、归一化、循环状态与来源映射
genedynamics/solvers/single/acp/                  策略、动作/柔顺数学与来源映射
genedynamics/learning/baselines/fada/             目标记录窗口、loss、IDM 适应
genedynamics/solvers/common/policy_execution.py   public observation → policy → task controller
scripts/baselines/{sources,environment,qualify_core}.py
.runtime/baseline_core_torch/                    独立 venv、安装证据与缓存
```

本地移植代码可在没有 `baselines/` 的情况下推理；上游 checkout 只在源码
对齐和原生复现时使用。大型资产、LFS 权重及外部 submodule 不随源码工具自动
取得。`fetch` 遇到已有 checkout 只核验，不覆盖本地修改，不切到新的 main。

| 来源 | 固定 commit |
| --- | --- |
| [FADA](https://github.com/LeCAR-Lab/FADA) | `290b3a9c5c0ebe2e9851b61350617544be05e9bc` |
| [Human2Sim2Robot](https://github.com/tylerlum/human2sim2robot) | `894eae2ec3ae39a573b81bd1860d14cc6bdfa6df` |
| [ACP](https://github.com/yifan-hou/adaptive_compliance_policy) | `74d68eca3bfb5cb59da131c6351d9e1c9b129ff6` |
| [Play2Perfect](https://github.com/kushal2000/play2perfect) | `ff2fc62873f6c6cd93164123c8358e9ce2d65c9b` |

许可证和移植路径见根目录 `THIRD_PARTY_NOTICES.md`，逐文件来源见各方法目录
中的 `source_manifest.json` 或 `SOURCE.json`。

## 3. 可装可删的核心验证环境

```bash
python3 -m scripts.baselines.sources fetch
python3 -m scripts.baselines.sources verify
python3 -m scripts.baselines.environment create --name baseline_core_torch --device cuda --gpu 0
python3 -m scripts.baselines.environment doctor --name baseline_core_torch --device cuda --gpu 0
```

CPU 机器使用另一个名字和 `--device cpu`。profile 的直接版本固定在
`requirements/baselines/core-pytorch.txt`；Torch 单独固定 2.5.1，CUDA 版本
使用 cu124。安装报告包含解析 wheel 的哈希，`requirements.lock.txt` 保存
实际完整依赖闭包；它不是作者全部原生训练环境的 lock。

本机 CUDA doctor 已实际运行 Linear/AdamW 前后向更新，设备为 RTX 4500 Ada
的 `cuda:0`。这个检查只证明 PyTorch CUDA 数值环境可用，不认证 MuJoCo/MJX
物理等价。JAX 在该 profile 中使用 CPU，P2 的 JAX CUDA 环境保留在原目录。

删除命令只删除匹配本仓库所有权 marker 的环境，源码与结果保留：

```bash
python3 -m scripts.baselines.environment clean --name baseline_core_torch --device cuda
```

原生 H2S2R README 的训练栈是 Python 3.8、IsaacGym Preview 4 和 Torch 2.1.2
cu118，不能直接安装进要求 Python ≥3.10 的项目环境。其 assets/data/pretrained
模型需经作者资源入口取得。ACP 编码器还依赖 `multimodal_representation`
submodule 和视觉/力资产，实际控制依赖外部 C++ 控制栈；核心环境未冒充这些资源。

## 4. 源码对齐与结果

```bash
.runtime/baseline_core_torch/venv/bin/python -m scripts.baselines.qualify_core \
  --device cpu --output docs/development/cross-ebdm-experiment-reports/p3_source_parity_20261014/cpu
.runtime/baseline_core_torch/venv/bin/python -m scripts.baselines.qualify_core \
  --device cuda --gpu 0 --output docs/development/cross-ebdm-experiment-reports/p3_source_parity_20261014/cuda
```

入口先核对固定源码，再生成测试日志、JUnit、资格 JSON、来源 lock、实际依赖
闭包和本地文件 SHA。缺源码、跳过测试或失败均不标通过。对齐范围包括：

- FADA：Planner/IDM 输出及输入梯度、action-history 对齐、冻结 planner、LoRA
  AdamW 更新、部署合并与归一化/checkpoint。
- H2S2R：actor/value、统计量、MLP/LSTM/GRU、checkpoint、Gaussian RNG、动作
  变换及分批历史 reset。
- ACP：virtual-target/pose/compliance 数学、原生 sparse diffusion 输出、loss、
  更新及 checkpoint；核心测试的合成编码器不作为原生视觉编码器资格。
- 公共 bridge：明确筛选 public observation，记录真实执行动作、终止及异常，
  完整 episode 始终有 `len(states) == len(actions) + 1`。

最初源码接入版本 **CPU 40/40、CUDA 40/40，均零跳过、零失败**：
[CPU 资格证据](cross-ebdm-experiment-reports/p3_source_parity_20261014/cpu/qualification.json)、
[CUDA 资格证据](cross-ebdm-experiment-reports/p3_source_parity_20261014/cuda/qualification.json)。
三方法分别为 9、15、12 项，公共执行 bridge 4 项。

首次 CUDA 完整运行的设备别名断言失败与相应 CPU 运行保留在同目录
`*_pre_device_alias_fix/`，修复仅解析测试中 `cuda` 的逻辑设备 index。
FADA 的上游 merged Transformer fastpath 与 unmerged LoRA 在 CUDA 上自身存在
约 `1.95e-4` 的差异；对相同路径的上游/本地比较完全一致，另外关闭 fastpath
验证合并等价，未通过放宽容差掩盖差异。

同设备上游/本地对齐不替代 CPU/GPU 物理后端资格，也不证明四任务的控制
接口已正确。具体范围与文件指纹以本次 `qualification.json` 为准。

## 5. solver 接口

```python
from genedynamics.solvers import FADASolver, Human2Sim2RobotSolver, ACPSolver
```

这三个属性按需加载，并注册为 `fada_torch`、`human2sim2robot_torch`、
`acp_torch`。它们是闭环策略接口，不伪装成采样 planner；不在 `act` 中训练。
项目顶层 solver 仍会加载现有 JAX stack，所以原生训练环境应独立进程运行。

FADA 的 `FADASolver.from_checkpoint(payload, device="cuda:0")` 接受作者
checkpoint mapping。`act` 接收 batched 的 `history_obs`、`history_act`、
`current_command`、`history_valid_mask`，按 checkpoint 归一化；历史右对齐，
每个观察对应之前实际执行的 action。`reset()` 清除 open-loop prefix。

H2S2R 构造时明确 `network_config`、`observation_shape`、`action_low/high`；
`restore(path_or_mapping)` 加载作者 checkpoint，`act(public_obs)` 返回实际动作，
`reset(dones=...)` 清除指定 episode 的循环状态。`native_lstm_config()` 提供
作者原生配置；数字孪生/reward/示范仍属于原生训练阶段。

ACP 构造使用源等价的 `ACPDiffusionPolicy`、实际编码器和 diffusion scheduler。
`act` 接收 `sparse` 公共传感历史与 `current_tool_pose_world`，返回世界系
reference/virtual-target pose、tool-frame stiffness、waypoint 时间等结构化命令。
任务 adapter 实现实际控制与观测采样；不能把输出压成固定刚度后仍称完整 ACP。

`PolicyExecutionController` 的 `observe(state)` 由任务所有，禁止把含隐藏真值
的 `ContactState.physics` 直接输入策略；`execute(state, request)` 必须返回
`ExecutedStep`，其中包含下一状态、实际施加动作、标量 done 和日志。当前 bridge
支持单 episode；batched evaluator 的 partial reset/重排须独立资格。

## 6. 后续资格与比较

1. 根据新执行记录逐路关闭原生资产、闭环稳定性和任务 source-support gate；
   H2S2R 资产仍需作者下载资源，其他未受阻路线继续独立推进。
2. 新接口/训练检查与源码 parity 完成后，充分训练并检查 source held-out sampling、
   源闭环、学习曲线、实际预算，再作为强基线进入 P4/P5。
3. ours 也迁移到相同公共 issued reference/history，避免旧 P2 carry 权限差异；
   所有失败、重放、shadow 和目标交互保持完整计费。
4. 四任务环境可达资格不等于四任务近期基线比较完成；T4 近期方法训练、原生系统
   待验事项继续遵循主路线图，论文只纳入最终冻结的正式证据。
