# PegInsert：执行环境修复与统一协议重跑 Runbook

首次审计日期：2026-09-13；clean-room P0--P1 冻结日期：2026-09-24。
适用范围：`results/arm/peg_insert` 下 8 方法 × 3 suite × 10 seed 的正式结果。

> **当前状态（以本段为准）**：旧的本地正式结果、训练产物、development
> 结果和 RunPod 快照已经归档到
> `results/_archive/peg_insert_protocol_reset_20260924/`。活动结果目录已经清空，
> policy 统一写入 `results/arm/peg_insert/_policies/`。P1 已固定 64-step、
> 64-sample、ID/Pose-OOD/Sensing-OOD 协议和 240-run 正式矩阵；两条 policy
> 已从零训练并冻结。P4 的 learned-reliability candidate 没有显示稳定的控制增益，
> 因此 canonical MGA 已冻结为 core-only：保留 RL prior、model-based rollout、
> manifold/LRC、revalidation、incumbent 与 emergency，不加载 learned reliability。
> 正式协议状态为 `frozen_core_only`，P5 可以写入 canonical 结果目录。
> 下文对 300-run 旧矩阵的分析只作为历史问题记录，不再描述活动协议。

### P0--P1 实施状态

| 阶段 | 状态 | 已冻结内容 |
|---|---|---|
| P0 clean-room | **完成** | 本机旧结果、训练产物、development、旧报告和 RunPod 快照全部进入 `results/_archive/peg_insert_protocol_reset_20260924/`；活动根目录不含任何旧 run 或 checkpoint |
| P1 protocol/runner | **完成** | 8 方法 × 3 suites × 10 seeds；64 steps、64 samples；全窗口 SSR；nominal-model / hidden-execution 边界；失败可见且 summary 可增量重建 |
| P2 policy training | **完成** | raw-action PPO 与 ATACOM tangent PPO 均从零训练 200k steps，以 seeds 104/105 的独立 6-run health gate 冻结 |
| P3 MGA core | **完成** | seeds 108/109 × 3 suites × {Model-based Only, MGA core, ATACOM} 的 18-run 配对矩阵通过；MGA core 6/6 insertion、5/6 full-window safe、0 jam，并严格优于 ATACOM 的 4/6 safe |
| P4 reliability | **完成（不晋级）** | 106+112 fit / 107+113 calibration 已产生未晋级 candidate；calibration AUROC 为 0.708/0.974/1.000，但已完成配对中 SSR 与 core 相同且 tail risk 有升有降，因此 candidate 保留为 development evidence，不进入 canonical MGA |

活动协议状态是 `frozen_core_only`。formal audit 已确认 8 方法 × 3 suites × 10 seeds
仍为 240 runs，正式 learned-component lock 仅包含已冻结的 raw PPO 与 ATACOM PPO。
P4 candidate 不在正式 lock 中，也不会被 `main/mga.yaml`、`no_rl_prior.yaml` 或
`no_retraction.yaml` 加载。
归档目录受 `results/` 的版本控制忽略规则保护，是本机 provenance，不会随普通
Git commit 自动上传。

### P2 冻结记录（2026-09-24）

| 组件 | 训练契约 | selected step | 独立开发验证 | SHA256 |
|---|---|---:|---|---|
| raw-action PPO | seed 0；200k；4 domains；4 envs；16×16；action 13 | 200000 | seeds 104/105 × ID/Pose-OOD/Sensing-OOD：6/6 insertion，0/6 full-window safe | `14f1b13176c871927b216e969c5251199e4f8bab0d4c00115cce333547ae137c` |
| ATACOM tangent PPO | seed 0；200k；4 domains；4 envs；16×16；action 6；Kc=1；tangent/control bound=0.58 | 200000 | seeds 104/105 × ID/Pose-OOD/Sensing-OOD：6/6 insertion，4/6 full-window safe，0/6 jam；mean nCVaR95=0.904 | `39378bad6d0698c8befb1ca58fbe983fd1d633b91a000d6303acd89210a46fff` |

ATACOM 的冻结验证位于
`results/_development/peg_insert_p2_atacom_final_validation/`。checkpoint 内保存的
`evaluated_checkpoint_sha256` 是写入验证 metadata 前的参数文件 hash `fbb20028...f9a42a`；
上表和 `_base.yaml` 锁定的是写入该 metadata 后的最终文件 hash。action=1.0 的失败版
和未被选择的 bound=0.55 版均保留在 clean-room archive 中，不参与正式结果。

Docker 的 15.6 GiB 内存无法同时保留 PPO 梯度图和 inline evaluator；第一次尝试
在首次编译阶段以 exit 137 终止，且未产生 policy。正式 P2 因而保留完全相同的
训练分布和预算，使用 `--cpu-low-memory --skip-inline-eval`，随后通过统一 runner
独立验证并把验证根目录、原始被验证 hash 与聚合指标写入 checkpoint metadata。
raw-action PPO 的 `0/6 safe` 是有意保留的纯 RL 边界：它提供有任务能力但不保证
安全的 proposal。ATACOM 则使用动作可控的 wrench viability chart、独立的 tangent /
control bounds，以及成功后的安全终端流形，从修复前的 0/6 safe、6/6 jam 提升到
4/6 safe、0/6 jam。两条 OOD sensing 轨迹仍各有一次轻微越界，说明名义 ATACOM
模型不等于绝对 OOD 保证；P3 先验证 MGA 的 model-based revalidation 能否降低这一
失配；P4 对 learned reliability 的独立验证没有产生稳定增量，因此不能把 ATACOM 或
learned gate 描述成绝对 OOD 安全保证。

### P3 执行契约：验证 MGA core

P3 不训练或加载 learned reliability，也不改动 P2 的两个冻结 policy。其目的，是先
回答安全改善是否来自 MGA 的 nominal model rollout、realization-aware refinement、
执行前 revalidation、receding incumbent 与 task-owned emergency，而不是来自一个尚未
校准的分类器。P4 仅作为独立 development study 检验 learned reliability 是否提供
额外增益；它不是 MGA core 安全机制成立的前提。

P3 使用未参与 policy 选择、P4 拟合/校准和正式评估的 development seeds 108/109，
在 `id_wide`、`ood_pose`、`ood_sensing` 上配对运行：

- `baseline/model_based_only`：没有 RL proposal 的 model-based 下界；
- `ablation/no_learned_reliability`：带冻结 raw PPO prior 的 MGA core；
- `baseline/atacom`：P2 已冻结的安全 RL baseline，用同 seed 重跑以保证配对比较。

全部 18 个 run 只能写到
`results/_development/peg_insert_p3_core/`。P3 的晋级门槛预先固定为：MGA core
6/6 insertion、0/6 jam、至少 5/6 full-window safe；其配对 safe success 必须严格高于
ATACOM，aggregate nCVaR95 不高于 ATACOM，并且不能靠完全拒绝 RL prior 退化为 MBO。
所有 force/torque、revalidation、incumbent、emergency 与 prior-selection diagnostics
必须有限且可重建。运行时间单独报告，不允许为了加速改变 64-step、64-sample 协议。

P3 未通过时，只允许修改 PegInsert task-owned contract 或 PegInsert 配置，并重新跑完整
18-run development matrix；不得改 scanning/humanoid，不得覆写 P2 checkpoint，也不得
提前写 canonical formal results。

### P3 验收记录（2026-09-24）

权威 development 根目录为 `results/_development/peg_insert_p3_core/`。18/18 个
`results.json` 与对应 64-step trajectory 均存在；两个冻结 policy 的 SHA256 与 P2 lock
一致，三个方法在 Sensing-OOD 上都实际观测到
`max|measured_lateral_force-lateral_force|=0.7071 N`。

| 方法 | ID safe | Pose-OOD safe | Sensing-OOD safe | 总 insertion | 总 jam | mean nCVaR95 | mean CPU planning time |
|---|---:|---:|---:|---:|---:|---:|---:|
| MGA core（w/o learned reliability） | 2/2 | 1/2 | 2/2 | 6/6 | 0/6 | **0.816** | 940 s |
| ATACOM | 2/2 | 2/2 | 0/2 | 6/6 | 0/6 | 0.930 | 81 s |
| Model-based Only | 0/2 | 0/2 | 0/2 | 6/6 | 6/6 | 1.011 | 612 s |

MGA core 达到预注册的 6/6 insertion、至少 5/6 full-window safe、0 jam，并以
5/6 对 4/6 的 paired safe success 严格超过 ATACOM，aggregate nCVaR95 也更低。
MBO 的 6/6 prefix insertion 随后全部 jam，说明 model-based rollout 单独不足以维持
完成后的安全状态。MGA 的六条轨迹都实际选择过 additive RL prior；384/384 个最终动作
通过 model revalidation，所有轨迹都有 unsafe refined/incumbent rejection，且没有
`emergency_unrecoverable`，因此结果不是退化成 MBO 或完全拒绝 prior 得到的。

唯一的 MGA core 失败是 Pose-OOD seed108 在 64 个物理步中出现一次越界
（violation rate 1/64，nCVaR95=0.953）。P4 随后检验 learned reliability 是否能稳定
降低这类残余 model-mismatch failure；结果未形成稳定增益，因此最终冻结 core-only。

CPU 上连续运行多个 MGA/MBO suite 会因 XLA 缓存累积出现 exit 137；缺失项均用新的
单-run 容器、同一 config/checkpoint/source 重新执行，而不是降低 samples/steps。
MGA 的平均 planning time 仍显著高于 MBO/ATACOM，是独立的性能优化项；按照既定路线，
它不阻塞 P4 的正确性校准，但必须在正式大矩阵前决定使用 GPU 或完成语义保持的编译优化。

### P4 执行记录：learned reliability development study

P4 保持 P3 已通过的 44 维 PegInsert reliability feature contract 和完整 MGA
控制路径不变；不迁移 RunPod 旧分支的 57 维 counterfactual scorer，也不修改 sampler、
refinement、revalidation、incumbent 或 emergency。训练数据由现有 unified runner 的
`results.json + trajectory.json` 转换：动作取已提交的 action，pose / wrench / violation /
jam 标签取 hidden execution environment 已持久化的 task signals，禁止在 nominal model
中 replay 后冒充 OOD 标签。

数据与评估 split 固定为：

- fit seeds：106、112；
- split-conformal calibration seeds：107、113；
- independent paired validation seeds：110、111；
- formal evaluation seeds：0--9，禁止进入以上任何 split；
- development suites：`id_wide`、`ood_pose`、`torque_stress`、`jam_stress`。

采集配置仅位于
`configs/arm/peg_insert/validation/reliability_collection.yaml`，继承 P3 的
`no_learned_reliability` MGA core，并只写入
`results/_development/peg_insert_p4_reliability_collection/`。Trainer 现在会拒绝
train/calibration 重叠、formal/validation seed 泄漏、policy hash 不一致、缺失或非有限的
execution signals，以及任一 binary risk head 缺少正类或负类。拟合产物首先必须是
`performance_validated=false` / `promotion_eligible=false` 的 development candidate。
该 candidate 最终没有晋级；正式协议通过冻结 core-only 解锁，而不是通过写入
`results/arm/peg_insert/_policies/reliability.json` 解锁。

seed106 的 MGA-core 首轮探针已完成，4/4 runs 正常退出且没有 OOM。252 个 committed-
horizon rows 中，force 为 3 positive / 249 negative，torque 为 56 / 196，jam 为
0 / 252。`ood_pose` 提供 force 正类，`torque_stress` 提供 torque 正类；原
`jam_stress` 被 MGA core 安全解决，因此它证明 controller 有效，却不能训练 jam head。

P4 不通过换成 MBO baseline 或降低正式 controller 的安全性去制造 jam。此前启动的 MBO
probe 在产出 `results.json` 前已停止，唯一的临时 manifest 也已删除，不参与任何 split。
补充配置 `validation/reliability_no_prior_collection.yaml` 使用因果 `no_rl_prior`
ablation；由于 P4 尚未产生 learned gate，采集时只关闭这个不存在的组件，仍保留同一
MGA refinement、revalidation、receding incumbent 和 task-owned recovery。

seed106 的正确三源分布已经通过：MGA core 252 rows，no-prior 63 rows，冻结 raw-PPO
prior 63 rows；合计 378 rows，force/torque/jam 的 positive/negative 分别为
6/372、63/315、4/374。seed107 的独立 calibration probe 也已通过同一契约：合计
378 rows，三类 positive/negative 分别为 6/372、64/314、4/374。这里 no-prior
表示没有 RL proposal 的同机制候选族，raw PPO 表示 MGA 实际需要筛选的 prior
candidate，而不是把 standalone RL 当成安全 controller 训练目标。Trainer 记录每行的
behavior name/method 与 policy hash。seeds112/113 已按相同三源协议完成；split 固定为
106+112 fit、107+113 calibration，不能依据后续 paired validation 结果重新选择数据或
模型。

四个 split seeds 已全部完成，冻结后的 development candidate 位于
`results/_development/peg_insert_p4_reliability/reliability_candidate.json`，SHA256 为
`7a809dbd0dd673495df30d0b6e402b189bf4d5640e1c0cc4088213124ffee325`。fit 与
calibration 各 756 rows；calibration 的 force/torque/jam AUROC 分别为
0.708/0.974/1.000，Brier 分别为 0.0285/0.0562/近零，joint-support rate 为
95.9%。该文件仍保持 `performance_validated=false`、`promotion_eligible=false`。

paired validation 使用 `validation/reliability_gate.yaml`。为了不伪造 checkpoint 的
晋级元数据，也不把最终 `model_based` OOD 规则换成更保守但不同的 `veto` 算法，runner
提供一个严格受限的 `reliability_validation_authoritative` 开关：它只允许
PegInsert MGA、`run_class=development`、`formal_experiment=false` 和未晋级的
development-fit candidate；正式 MGA 默认关闭并仍要求 promotion metadata。验证使用
seeds110/111，逐 suite 与 `ablation/no_learned_reliability` 配对。

冻结决策时共得到 9/12 个结果，其中形成 4 个完整配对：ID seeds110/111、Pose-OOD
seed110、Sensing-OOD seed110。四对的 insertion 与 full-window SSR 完全相同；ID 的
gate/core nCVaR95 分别为 0.775/0.767 和 0.753/0.766，Pose-OOD 为
0.929/0.929，Sensing-OOD 为 0.789/0.811，表现有升有降。Pose-OOD seed110 中 gate
因 joint support 外而 abstain，执行轨迹与 core 完全相同。鉴于没有安全成功率增益、
tail risk 无一致方向且 learned path 增加训练/校准/运行复杂度，剩余 3 个诊断 run 被
停止；已有数据和 candidate 全部保留，不删除、不进入正式统计。

相关文档：[三页图表与叙事安排](paper_figures_README.md)、[PegInsert 出图草稿（已过期，重跑后需重写）](peg_insert_figure_draft.md)、[完整工程历史](experiment_run_plan.md)。

---

## 0. 历史问题：为什么必须重跑

P0 已归档的旧 300 个 run 有三处使结果不可直接比较的问题：

| 编号 | 问题 | 影响面 |
|---|---|---|
| **P1** | `standalone_rl` / `issa` / `atacom` / `pegasusflow` 的**执行环境没有生效**，整条 rollout 跑在名义 planning env 上 | 这 4 个方法的 `ood_pose` / `ood_sensing` 共 80 个 run 全部无效 |
| **P2** | `main/mga` 与 `ablation/no_rl_prior` 的 **ID 与 OOD 是两套不同算法配置** | MGA / no-prior 行的跨 setting 比较混入了算法变化 |
| **P3** | 指标在**执行环境的 socket 坐标系**下计算，但 P1 涉及的方法物理跑在名义 socket 上 | 与 P1 同源，修好 P1 后自动一致 |

修复 P1 + P2 后需要**整套重跑**，原因见 [§4.6 设备一致性](#46-设备一致性不可混跑)。

### 0.1 复现这三个问题的历史证据

以下命令针对归档前的目录布局保留，仅用于解释 reset 的原因；不要对新的活动
目录运行，也不能作为当前协议的验收命令。

```bash
# P1：这 4 个方法的 ood_sensing 与 id_wide 应当逐点不同；当前 10/10 seed 完全相同，
#     且 measured-true 恒为 0（应为 0.707 N）
python3 - <<'PY'
import json, numpy as np
root='results/arm/peg_insert'
M=['baseline/standalone_rl','baseline/issa','baseline/atacom','baseline/mppi','baseline/dial',
   'baseline/pegasusflow','baseline/model_based_only','ablation/no_rl_prior',
   'ablation/no_learned_reliability','main/mga']
print(f"{'method':34s}{'max|meas-true|':>16s}{'id==sensing':>14s}")
for m in M:
    d=[]; eq=[]
    for s in range(10):
        t={k: json.load(open(f'{root}/{m}/level_{k}/seed_{s}/trajectory/trajectory.json'))['task_signals']
           for k in ('id_wide','ood_sensing')}
        ts=t['ood_sensing']
        d.append(np.abs(np.array(ts['measured_lateral_force'])-np.array(ts['lateral_force'])).max())
        eq.append(np.allclose(t['id_wide']['lateral_force'], ts['lateral_force']))
    print(f"{m:34s}{np.max(d):16.4f}{sum(eq):>11d}/10")
PY

# P2：ID 与 OOD 的 method_params 差异
python3 - <<'PY'
import json
for m in ['main/mga','ablation/no_rl_prior']:
    a=json.load(open(f'results/arm/peg_insert/{m}/level_id_wide/seed_0/results.json'))['config_snapshot']
    b=json.load(open(f'results/arm/peg_insert/{m}/level_ood_sensing/seed_0/results.json'))['config_snapshot']
    print('###', m, a['metadata']['protocol'], '->', b['metadata']['protocol'])
    for k in sorted(set(a['method_params'])|set(b['method_params'])):
        if a['method_params'].get(k)!=b['method_params'].get(k):
            print(f'   {k:26s} ID={a["method_params"].get(k)!r}  OOD={b["method_params"].get(k)!r}')
PY
```

---

## 1. P1 已完成：修复 baseline 的执行环境

### 1.1 现状

环境插件本身是对的：[`_contact_task.py:81-93`](../../genedynamics/experiments/plugins/environments/_contact_task.py#L81-L93) 会在存在 `execution_env_params` 时额外编译一个执行环境并挂到 `env._experiment_execution_env`，[`:119 reset_state`](../../genedynamics/experiments/plugins/environments/_contact_task.py#L119) 也确实从执行环境 reset。

断点在方法工厂 [`contact_receding.py::_build_baseline_solver`](../../genedynamics/experiments/plugins/methods/contact_receding.py#L360-L450)：

| 方法 | 真实推进用的 env | 状态 |
|---|---|---|
| MPPI | `step_fn=execution_env.step`（[:378](../../genedynamics/experiments/plugins/methods/contact_receding.py#L378)） | ✅ 正确 |
| DIAL | 在 `METHOD_TABLE` 内，走 `make_mga`，执行环境正常传入 | ✅ 正确 |
| MGA / no_rl_prior / no_learned_reliability / MBO | 同上 | ✅ 正确 |
| **PegasusFlow** | 只收到 `env`（[:384](../../genedynamics/experiments/plugins/methods/contact_receding.py#L384)） | ❌ |
| **Standalone RL** | `RLPolicyController(env, …)`，[`:118 self.env.step`](../../genedynamics/solvers/common/rl_policy_controller.py#L118) | ❌ |
| **ISSA** | 内部构造 `RLPolicyController(env, …)`（[issa.py:41](../../genedynamics/solvers/single/issa/issa.py#L41)），且投影本身在 [`issa_jax.py:45`](../../genedynamics/solvers/single/issa/backends/issa_jax.py#L45) 复用 `self.env.step` 作为真实推进 | ❌ |
| **ATACOM** | `AtacomEnvWrapper(env)`，[`wrapper.py:71-74`](../../genedynamics/solvers/single/atacom/wrapper.py#L71-L74) 内部 `self.env.step` | ❌ |

[`contact_receding.py:504`](../../genedynamics/experiments/plugins/methods/contact_receding.py#L504) 的 `solver.execution_env = execution_env` 只被 [`mga_jax.py:245`](../../genedynamics/solvers/single/mga/backends/mga_jax.py#L245) 读取，对上述 4 个方法是死代码。

### 1.2 统一的语义约定

修复必须对所有方法采用**同一条语义**，否则 OOD 的含义在方法之间不一致：

> **模型知识（planner rollout、约束流形、安全指标）留在名义 planning env；
> 真实推进与观测来自 execution env。**

对照现有实现，这正是 MPPI 与 DIAL 已经在做的事（[`receding_horizon.py:244-245`](../../genedynamics/solvers/common/receding_horizon.py#L244-L245) 只用 `step_fn` 做真实推进，planner 的采样 rollout 仍绑在 `solver.dynamics`）。

按此约定，四个方法的改法：

**(a) PegasusFlow — 1 行，与 MPPI 完全对称**

链路已经通了，只差最后一棒：`PegasusFlowSolver` 的构造函数已接受 `step_fn`（[pegasusflow.py:39](../../genedynamics/solvers/single/pegasusflow/pegasusflow.py#L39)），后端会优先采用它（[pegasusflow_jax.py:174](../../genedynamics/solvers/single/pegasusflow/backends/pegasusflow_jax.py#L174)，否则回落到 `build_brax_step(model_env)`），`make_controller` 也已转交给 `RecedingHorizonController`（[pegasusflow.py:62-66](../../genedynamics/solvers/single/pegasusflow/pegasusflow.py#L62-L66)）。

在 [`contact_receding.py:384`](../../genedynamics/experiments/plugins/methods/contact_receding.py#L384) 的 `PegasusFlowSolver(...)` 调用里补上与 MPPI 同形的参数：
`step_fn=(execution_env.step if execution_env is not None else None)`。

**(b) Standalone RL — 直接换 env**

RL 是纯反应式方法，不存在「模型 vs 执行」之分：策略的观测**应该**来自执行环境（sensing 偏置必须真正喂进策略，这正是 Sensing-OOD 要测的东西）。最小改动是构造时用 `execution_env or env`。注意 [`rl_policy_controller.py:96-98`](../../genedynamics/solvers/common/rl_policy_controller.py#L96-L98) 与 [`:66`](../../genedynamics/solvers/common/rl_policy_controller.py#L66) 两条路径都引用 `self.env.step`，换 env 即可两条同时生效。

**(c) ISSA / ATACOM — 需要「投影用名义、推进用执行」的拆分**

这两个方法的安全机制本身就是它们的贡献，其模型知识应留在名义 env：
- ISSA 的 AdamBA 在 [`issa_jax.py:42-46`](../../genedynamics/solvers/single/issa/backends/issa_jax.py#L42-L46) 用 `env.safety_index` + `env.step` 搜索安全动作，然后在 [`:266 project_and_step_with_info`](../../genedynamics/solvers/single/issa/backends/issa_jax.py#L266) **复用该 transition 当作真实推进**。
- ATACOM 在 [`wrapper.py:71-74`](../../genedynamics/solvers/single/atacom/wrapper.py#L71-L74) 用流形算出 `u` 后立刻 `self.env.step`。

两个可选改法：

| 选项 | 做法 | 代价 |
|---|---|---|
| **C1（推荐）** | 给投影/wrapper 增加一个独立的 `step_env`：搜索/流形仍用名义 env，最终推进改用执行 env | 改动略大，但语义与 MPPI/DIAL/PegasusFlow 完全一致 |
| C2 | 停止复用投影 transition，让 `RLPolicyController` 用执行 env 单独推进一步 | 改动最小，但会多编译一个 `env.step`。[`rl_policy_controller.py:89-93`](../../genedynamics/solvers/common/rl_policy_controller.py#L89-L93) 的注释说明当初融合就是为了省 16 GB Mac 的内存——**在 GPU 上这个顾虑基本消失**，所以 C2 在本次重跑里是可接受的 |

选哪个都行，但**必须在配置或 `component_contract` 里记录选择**，不要留成隐式行为。

### 1.3 验收

修复后重跑并检查（这是 §0.1 第一段脚本的反向断言）：

- 10 个方法在 `ood_sensing` 上的 `max|measured−true|` **全部**为 `0.7071`；
- `id_wide` 与 `ood_sensing` 的 `lateral_force` **没有任何方法**出现 10/10 相同；
- `ood_pose` 每个 seed 的 `lateral_error` 在 t=0→1 之间不再出现坐标系跳变。

---

## 2. P1 已完成：ID 与 OOD 统一到同一个算法

### 2.1 要统一成什么

保留 OOD 侧当前在用的那套机制（additive prior + joint-support reliability），把 ID 也切过去。四个差异项：

| 参数 | ID 当前（历史） | OOD 当前 | **统一后** |
|---|---|---|---|
| `prior_mode` | 未设（默认 guided） | `additive` | **`additive`** |
| `reliability_support_mode` | `state` | `joint` | **`joint`** |
| `reliability_ood_policy` | 未设（默认 `veto`） | `model_based` | **`model_based`** |
| `reliability_ckpt` | `peg_insert_cpu/_policies/reliability_q95.json` | `peg_insert/_policies/reliability_v2.json` | **`peg_insert/_policies/reliability.json`** |

### 2.2 能跑通的依据

- `prior_mode: additive` 在后端**没有任何 suite / level 分支**：[`mga_jax.py:1815 _replan_additive`](../../genedynamics/solvers/single/mga/backends/mga_jax.py#L1815)、[`:1912` 的分发](../../genedynamics/solvers/single/mga/backends/mga_jax.py#L1912) 只看 `prior_mode`。`reliability_support_mode` / `reliability_ood_policy` 同样是与 level 无关的开关（[`mga_jax.py:380-392`](../../genedynamics/solvers/single/mga/backends/mga_jax.py#L380-L392)）。
- reliability 模型本身**就在 ID 上训过**。`results/arm/peg_insert/_policies/reliability.json` 的 metadata：

  ```
  training_suites   {id_tight:196, id_wide:196, ood_compound:294, pareto_force15:196, pareto_force22:196}
  calibration_suites{id_tight:98,  id_wide:98,  ood_compound:147, pareto_force15:98,  pareto_force22:98}
  training_seeds [100,101]   calibration_seeds [102]   evaluation_seeds_excluded [0..9]   split_disjoint true
  ```
  ID 在其 support 内，不存在「只在 OOD 上标定过」的退化风险。

### 2.3 已知风险

1. **ID 从来没在这套配置下跑过。** `reports/mga/peg_insert/upgrade_v4/eval/summary.csv` 里只有 4 个组合（`{mga, no_rl_prior} × {ood_pose, ood_sensing}`），ID 是空白。
2. **有过 OOM。** [`probe_mga.log`](../../reports/mga/peg_insert/upgrade_v4/probe_mga.log) 结尾是 `RUN_EXIT_CODE=137`。并行度必须受控。
3. **不要假设 ID 还是 90%。** 该机制在 OOD 上的 prior 接受率是 0.66，历史 ID 配置是 0.40 —— 机制确实不同，ID 的数字会动。这本来就是统一协议的代价。
4. **checkpoint 文件名与 lock 不一致。** `configs/arm/peg_insert/main/mga.yaml` 写 `_policies/reliability.json`，而历史 OOD run 的 `provenance` 记录的是 `_policies/reliability_v2.json`。二者 **sha256 相同**（`45ea1b37bc99fd41192956b28d47c81fb0125689955da95840b2110ac7c96db1`），是同一个文件改过名。统一后只保留 `reliability.json` 这一个路径。

### 2.4 探针（固定版本之前必须做）

用**非评估 seed**、写到开发目录，不污染正式结果：

```bash
python -m genedynamics.experiments.runner configs/arm/peg_insert/main/mga.yaml \
  --seed 110 --suite id_wide \
  --development-root results/_development/peg_insert_probe --resume

python -m genedynamics.experiments.runner configs/arm/peg_insert/ablation/no_rl_prior.yaml \
  --seed 110 --suite id_wide \
  --development-root results/_development/peg_insert_probe --resume
```

参考耗时：当初同类探针（`mga ood_pose seed 110`，CPU）是 **10 分 42 秒**（含编译）。ID 只会更快。

探针通过的判据：
- 进程正常退出（不是 137）；
- `results.json` 的 `config_snapshot.method_params.prior_mode == "additive"`；
- `trajectory.json` 的 `infos[0]` 含 `additive_prior_selected` / `additive_prior_candidate_count`，**不含** `proposal_rl_weight`；
- `diagnostics.selected_revalidated_safe` 存在且有限；
- `provenance.checkpoints.reliability_ckpt.sha256 == 45ea1b37…`。

### 2.5 当前固定版本

探针通过后，把下列内容写进 `configs/arm/peg_insert/main/mga.yaml` 的 `method_params` 与 `metadata.learned_component_lock`，并让 `_base.yaml` 的 `metadata.protocol` 改成新名字：

```yaml
# protocol 名
metadata.protocol: mga_peg_insert_core_only

# 算法
controller_method:          mga_controllable_gate
prior_mode:                 additive
prior_lambda_shift:         0.8
prior_include_incumbent:    true
prior_trust_radius:         0.35
prior_stochastic_samples:   8
prior_atacom_samples:       0
prior_fallback_mode:        receding_incumbent
prior_acceptance:           true
prior_improvement_epsilon:  0.0
prior_risk_tolerance:       [0.0, 0.0, 0.0, 1.0]

policy_ckpt:      results/arm/peg_insert/_policies/shared_ppo_seed0.pkl
learned_reliability: false
reliability_ckpt: null
```

**命名约定**：协议名描述配置内容，不使用 `v2` / `v3` / `v5` / `paper_v5`
这类阶段编号。P4 candidate 只留在 development root，不复制成 canonical checkpoint。

`ablation/no_rl_prior.yaml` 继承 `main/mga.yaml`，只关闭 RL prior。
`ablation/no_learned_reliability.yaml` 现在只是保留历史入口的非正式 alias，不再是有效
消融，也不进入 paper runner。

---

## 3. 重跑矩阵与预算

### 3.1 当前正式矩阵

8 方法 × 3 suites × seeds 0–9 = **240 runs**。

```
main/mga                        ablation/no_rl_prior          ablation/no_retraction
baseline/issa                   baseline/atacom               baseline/mppi
baseline/dial                   baseline/pegasusflow
suites: id_wide, ood_pose, ood_sensing
```

`standalone_rl`、`model_based_only` 和旧名 `no_learned_reliability` 的配置保留为
diagnostic；后者与 canonical MGA 相同，不计入正式矩阵，也不由 paper runner 启动。

下面的单 run 时间来自归档旧协议，只能用于保守估算机器预算，不能与新协议结果
混合或作为 P1 后的算法性能数字。

### 3.2 实测单 run 耗时（当前 CPU：Mac14,7 / M2 / 8 核 / Docker 15.6 GiB）

| 方法 | ID | Pose-OOD | Sensing-OOD |
|---|---:|---:|---:|
| Standalone RL | 13 s | 11 s | 11 s |
| ATACOM | 13 s | 11 s | 11 s |
| PegasusFlow | 171 s | 152 s | 155 s |
| DIAL | 143 s | 325 s | 325 s |
| MPPI | 147 s | 319 s | 322 s |
| ISSA | 240 s | 235 s | 221 s |
| Model-based Only | 194 s | 356 s | 362 s |
| w/o RL prior | 208 s | 380 s | 371 s |
| core-only MGA（旧 w/o learned reliability） | 233 s | 773 s | 446 s |

以上来自归档协议，只用于预算；新正式 MGA 就是 core-only 路径，不再重复运行一条
`w/o learned reliability`。

### 3.3 预算

| 范围 | runs | CPU 串行 | CPU 4 进程（约 3× 有效） |
|---|---:|---:|---:|
| 现状（已花） | 300 | 21.5 h | — |
| 仅修 P1 | 80 | 2.2 h | ~45–60 min |
| 修 P1 + ID 统一 | 100 | 4.4 h | ~1.5–2 h |
| **整套重跑（本文档的目标）** | **300** | **23.2 h** | **~8–9 h** |

CPU 并行只按 3× 估：8 核里 JAX 本身已多线程，进程间抢核，而且已经 OOM 过一次；4 个 MJX 进程（batch 64）大约就是 16 GB 的上限。

### 3.4 GPU 预期

环境是 `backend="mjx"`（[`peg_insert_brax.py:332`](../../genedynamics/envs/domains/manipulation/peg_insert_brax.py#L332)），MJX 本就是为 GPU 写的，CPU 是它的慢路径。

推算依据（**不是实测**）：
- 一个 MGA run 的串行批量 step 数 ≈ `n_steps 64 × Ndiffuse 2 × (Hsample 8 × n_frames 10)` ≈ **1.1 万**次「batch-64 的 mjx.step」。CPU 实测 273 s ⇒ 约 25 ms/次。
- batch 64 远未吃满 GPU，单次批量 step 基本是 kernel-launch bound（典型 0.5–1.5 ms）⇒ 纯计算 6–17 s。
- 瓶颈会转移到 **JIT 编译**（GPU 上这个图编译更慢，每进程约 1–3 min）。因此必须用 `--seeds 0 1 … 9` 把 10 个 seed 放进**同一个进程**摊销编译。`ood_pose` 每个 seed 几何不同，很可能仍会重编译。

端到端保守预期 **5–10×**：整套 300 runs 从 ~23 h 降到 **2.5–5 h**。上 GPU 后第一件事就是量一个 run 来替换这个估计（见 §4.5）。

---

## 4. 在 GPU 上跑（不用 Docker）

### 4.1 前置条件

- NVIDIA 驱动支持 CUDA 12.x（`nvidia-smi` 能跑）
- Python 3.10（与既有结果的 `package_versions` 对齐；3.11 也可，但要在 provenance 里体现）
- 仓库本体 + `results/arm/peg_insert/_policies/shared_ppo_seed0.pkl` 与 `shared_atacom_ppo_seed0.pkl` 必须存在且 sha256 匹配；正式协议不需要 reliability checkpoint
- PegInsert 的 Panda 模型是**仓库内资产**（`genedynamics/envs/assets/franka_panda/panda_arm.xml`），**不需要** mujoco_menagerie

### 4.2 安装（native venv）

```bash
python3.10 -m venv ~/venvs/genedynamics
source ~/venvs/genedynamics/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 与 docker/install/install_gpu_train.sh 同源，去掉 PegInsert 用不到的 torch/3DGS 依赖
python -m pip install \
  "numpy>=1.26.0" "scipy>=1.13.0" "pyyaml>=6.0.1" "tqdm>=4.65.0" \
  "trimesh==4.10.1" "imageio>=2.37.2" "einops>=0.8.2" "jinja2>=3.1.0" \
  "matplotlib>=3.8.0" "seaborn>=0.13.0" "pandas>=2.2.0" \
  "gym>=0.26.2" "gymnasium>=0.29.1" \
  "mujoco>=3.1.3" "mujoco-mjx" "gin-config>=0.5.0" \
  "osqp>=0.6.7.post3" "pytest>=9.0.2"

python -m pip install -U "jax[cuda12]" "brax==0.14.1"

# 不要用 requirements.txt（它会拖进 torch / gsplat / lpips / pin 等 PegInsert 不需要的重依赖）
python -m pip install --no-deps -e /path/to/enerdynamics
```

`genedynamics` 里所有 `import torch` 都在 try/except 内（`core/backends/runtime/manager.py`、`core/prob/noise_sampler.py`、`core/constraints/core/array_interface.py`），PegInsert 路径不需要 torch。

### 4.3 环境变量

```bash
export JAX_PLATFORMS=cuda            # 没有 GPU 时直接报错，而不是悄悄退回 CPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85   # 单进程独占；并行多进程时按进程数下调
export MUJOCO_GL=egl                 # 渲染 GIF/PNG 用；无显示器时 egl 比 osmesa 快很多
export PYTHONUNBUFFERED=1
```

### 4.4 ⚠️ 必须改一处配置：`device`

`configs/arm/peg_insert/_base.yaml` 里是 `device: cpu`，它会经 [`experiment.py:184`](../../genedynamics/experiments/framework/experiment.py#L184) 传到 [`jax_backend.py:63-80`](../../genedynamics/core/backends/runtime/jax_backend.py#L63-L80)。**不改的话不会报错，只会打印 `Using CPU devices` 然后在 GPU 机器上慢慢跑 CPU。**

改为 `device: gpu`。注意该后端在找不到 GPU 时只打一条 Warning 就**静默回落 CPU**（[`jax_backend.py:66-70`](../../genedynamics/core/backends/runtime/jax_backend.py#L66-L70)），所以必须配合 §4.5 的断言。

同时注意：`device` 是 `config_snapshot` 的一部分，改了它之后 `--resume` 会把所有旧结果判为 config 不匹配而重跑 —— 本次正是要整套重跑，符合预期。

### 4.5 开跑前的两项断言

```bash
# 1) JAX 真的在 CUDA 上
python -c "import jax; print(jax.default_backend(), jax.devices()); assert jax.default_backend()=='gpu'"

# 2) MJX 能编译本任务的模型，并量一次真实耗时（替换 §3.4 的估算）
time python -m genedynamics.experiments.runner configs/arm/peg_insert/main/mga.yaml \
  --seed 110 --suite id_wide \
  --development-root results/_development/peg_insert_probe --resume
```

第 2 条同时就是 §2.4 的协议探针 —— 在 GPU 上只需要跑这一次，两个目的一起达成。
跑完检查日志里是 `Using GPU devices: [...]` 而不是 `Using CPU devices`。

### 4.6 设备一致性：不可混跑

CPU 与 GPU 的数值结果**不会一致**（XLA kernel 不同、float32 归约顺序不同）。因此：

> 一旦决定用 GPU，**240 个 run 必须全部在 GPU 上重跑**；不能保留任何一条 CPU 结果混进主表。

这是「整套重跑」而不是「只补 80 个」的根本原因。

### 4.7 执行

单 GPU、每个 config 一个进程、10 个 seed 同进程以摊销编译：

```bash
cd /path/to/enerdynamics
configs=(
  configs/arm/peg_insert/main/mga.yaml
  configs/arm/peg_insert/ablation/no_rl_prior.yaml
  configs/arm/peg_insert/ablation/no_retraction.yaml
  configs/arm/peg_insert/baseline/issa.yaml
  configs/arm/peg_insert/baseline/atacom.yaml
  configs/arm/peg_insert/baseline/mppi.yaml
  configs/arm/peg_insert/baseline/dial.yaml
  configs/arm/peg_insert/baseline/pegasusflow.yaml
)
for c in "${configs[@]}"; do
  for s in id_wide ood_pose ood_sensing; do
    python -m genedynamics.experiments.runner "$c" \
      --suite "$s" --seeds 0 1 2 3 4 5 6 7 8 9 --resume \
      2>&1 | tee -a reports/mga/peg_insert/rerun.log
  done
done
```

[`scripts/paper/mga/run_peg_insert.sh`](../../scripts/paper/mga/run_peg_insert.sh)
采用相同结构：每个方法--suite 组合使用一个新的 Python 进程。冻结协议不变，
同时可在 suite 切换时释放 JAX/XLA cache，避免 ID 与 OOD 编译留在同一进程时
叠加 CPU Docker 内存峰值，并保留按 suite 排查与断点续跑能力。

并行化建议：
- **先不要并行。** 单进程用满 `XLA_PYTHON_CLIENT_MEM_FRACTION=.85`，先量清楚单 run 真实耗时。
- 若确认显存有余（例如单进程峰值 < 40% VRAM），可开 2–3 个进程并行，并把 `MEM_FRACTION` 相应降到 `.3`。
- **不要**跨 GPU 分散同一次重跑，除非确认同型号同驱动 —— 否则又回到 §4.6 的混跑问题。

### 4.8 跑完的后处理

```bash
bash scripts/paper/mga/render_results.sh      # 重新渲染 PNG/GIF（vis 已按执行环境渲染）
bash scripts/paper/mga/summarize_results.sh   # -> reports/mga/peg_insert/{summary,paired_deltas,representative_seeds}
bash scripts/paper/mga/verify_results.sh      # audit + verify --require-visuals
```

注意这三个脚本覆盖 surface_scan / peg_insert / push_to_line 三个任务；若只重跑了 PegInsert，直接跑会因其余任务缺结果而失败。只做 PegInsert 时改用：

```bash
python -m genedynamics.experiments.utils.vis results/arm/peg_insert
python -m genedynamics.experiments.utils.metrics summarize results/arm/peg_insert --output reports/mga
python -m genedynamics.experiments.utils.metrics verify configs/arm/peg_insert --require-visuals
```

---

## 5. 归档与 provenance

### 5.1 P0 归档（已完成）

归档根目录为 `results/_archive/peg_insert_protocol_reset_20260924/`，其中
`local/` 保存本机旧正式结果和训练产物，`local_development/` 保存历史升级实验，
`runpod_snapshot/` 保存 RunPod 正式结果、训练结果、development 与其既有归档，
`derived_reports/` 保存旧报告。完整来源、文件数和 checkpoint hash 见归档内
`archive_manifest.json`。活动目录不从归档自动 resume 或聚合。

### 5.2 重跑后重写 manifest

归档中的 `local/paper_composite/paper_result_manifest.json` 描述的是「历史
ID/baseline + 升级后的 OOD」这种拼接来源，已经整体失效。新 manifest 应记录：

- `protocol: mga_peg_insert_cleanroom_p1`（所有 240 个 run 同一个值）
- `device` 与 `package_versions`（GPU 机器上的实际值）
- 两个 checkpoint 的路径 + sha256
- P1 修复所选的方案（ISSA/ATACOM 用 C1 还是 C2）
- `seeds: 0..9`、`expected_formal_task_runs: 240`
- 归档路径 `results/_archive/peg_insert_protocol_reset_20260924/`

### 5.3 需要同步更新的文档

- [`peg_insert_figure_draft.md`](peg_insert_figure_draft.md) —— 其中「记录全部是 guided 配置」的说法在统一协议后彻底失效，需重写。
- [`paper_figures_README.md` §5.1](paper_figures_README.md) —— 该节的数据源表与「composite 来源」描述需替换为单一协议。
- `latex/latex_mga/tex/peg_insert_results.tex`、`sections/exp.tex` —— 全部数值与结论句需按新结果重写。
- `latex/latex_mga/tex/algorithm.tex` —— 当前描述的是 guided structured-bank 形式，与 `prior_mode: additive` 不符，需要作者决定如何改写。
- [`latex/latex_mga/figures/envs/extract_frames.py`](../../latex/latex_mga/figures/envs/extract_frames.py) —— PegInsert 面板目前取自已废弃的 `results/arm/peg_insert_cpu/formal_canonical_v4/gifs/id_wide__full_mdac__seed12.gif`（`seed12` 甚至不在评估 seeds 0–9 内），应改指向重跑后的正式目录。

---

## 6. 验收清单

重跑完成后，下列每一条都必须为真才能用于论文：

- [ ] 240 个 `results.json` 全部存在，且 `config_snapshot.metadata.protocol` 全部等于 `mga_peg_insert_cleanroom_p1`
- [ ] 240 个 run 的 `config_snapshot.device` 一致（全 `gpu` 或全 `cpu`）
- [ ] `provenance.package_versions` 在 240 个 run 之间一致
- [ ] 三个活动 checkpoint 的 sha256 在适用的 run 之间一致
- [ ] `ood_sensing` 下所有 8 个正式方法均执行真实 sensing bias，而不是名义环境
- [ ] 任何方法的 `id_wide` 与 `ood_sensing` 轨迹**都不再**逐点相同
- [ ] `main/mga` 与 `ablation/*` 的 `method_params.prior_mode` 三个 suite 一致，均为 `additive`
- [ ] 每个 seed 目录都有 `trajectory/trajectory.json` + `trajectory_best.png` + `trajectory_best.gif`
- [ ] `metrics verify --require-visuals` 通过
- [ ] 新的 `paper_result_manifest.json` 已写入并与实际落盘一致
