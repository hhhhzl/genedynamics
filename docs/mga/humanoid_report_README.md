# H1 Push-to-Line：P1–P4 论文报告缺口与修复顺序

最近审计：2026-09-15。本文依据落盘结果、当前代码/config 和 [experiment_run_plan.md](experiment_run_plan.md) 整理状态。这里的 P1–P4 是四类 H1 子任务，与工程实施计划 P0–P12 的编号不同。下方“当前快照”覆盖 2026-09-12 的旧审计文字；旧结果仍保留用于说明修复来源，不能替代冻结版本复验。

三页图表布局见 [paper_figures_README.md](paper_figures_README.md)。

## 0. 2026-09-15 当前快照：A–F 到底完成了多少

这里的“完成”分三层：**代码/契约完成**、**开发 seed 验收通过**、**冻结后正式统计完成**。只有第三层可以直接支撑论文统计结论。当前总体状态是：A 完成；B 是正在执行的 active gate；E/F 先完成 P1–P3 子阶段；C/D 与 P4 的 E/F 子阶段暂停，等待 P1–P3 正式就绪。

| 阶段 | 当前状态 | 已经完成且有证据的部分 | 仍缺少的晋级条件 |
|---|---|---|---|
| A. 任务与报告契约 | **完成（代码/开发契约）** | P1 使用 force-step pass，不再用固定箱体的 line-goal SSR；P2/P3/P4 的成功与安全口径分开；4 ms 物理子步、真实完成/中止长度、padding 排除和 abort 原因均落盘；P1/P3 隐藏目标线，P2 marker 对齐目标箱体姿态的前沿 | 正式冻结仍归 F；A 本身不等于算法性能通过 |
| B. P1/P2 保留与 P3 复验 | **部分完成** | P2 nominal/OOD 的当前 model-based、full-MGA 与 no-prior 开发结果均已完整落盘；P3 的 MBO/full-MGA 双 seed 均有安全成功轨迹和真实 GIF | P1 的当前同冻结 full-MGA 只有 15 N 结果，seed111 暴露一个末端 4 ms 力带反例，30 N 在 CPU 预算内尚无结果；P3 的 no-RL-prior 与 geometry ablation 仍缺，早期 seed101/102 不能合成冻结 2/2 |
| C. P4 gait/prior/负载接触统一 | **部分完成** | DIAL 参考、direct joint-target realization、同步 startup、measured support/phase、任务时钟与低层动作映射已接入；有负载时能够安全推进到 body progress 约 0.204 m | 当前 canonical walk prior 尚未在现行 observation/action/certificate 下产生稳定的双脚安全闭环；旧 `p4_viability3` 的双脚轨迹依赖旧 learned prior、H8/node2 和更大 residual，不能直接晋级 |
| D. 111–132 步接续失败诊断 | **诊断完成；修复未完成** | 找到并修复 node/dense 表示和 task-owned recovery provenance 问题；UNLOAD minimum dwell、measured phase、roll/pitch recovery bank 均由同一模型重验；旧 111/124/128 边界已向后推进 | phase-lag rescue 仅把 abort 从 131 推到 132；下一状态仍无 recursively safe successor。局部“本 horizon 安全”尚未变成 terminally viable / recursively feasible 证书 |
| E. learned prior / reliability 更新 | **未通过** | 200k walk PPO 与 reliability 文件存在，schema/provenance 检查基础设施存在 | 现有 P4 walk checkpoint 在当前 realization 下没有证明优于 Gaussian；reliability 也未在新 P4 风险/horizon 上完成独立校准。当前结果不能标为有效 full-MGA |
| F. 两开发 seed → 冻结 → 正式补跑 | **P1–P3 预检进行中；P4 未开始** | seeds、方法、指标、统一入口、开发目录、报告/渲染/验证闭环已有规划和 dry-run 证据 | 先完成 P1/P3 的物理双 seed、八算法入口 smoke 和同冻结 provenance；之后才恢复 P4。P4 没有 full-MGA 2/2 strict-safe success 前不得并入正式 seeds 0–9 矩阵 |

**执行顺序纠正。** A–F 是工作包，不应机械地按字母推进。当前硬门槛是：先完成 A → B → E 中属于 P1–P3 的 learned-component 验证 → F 中属于 P1–P3 的正式运行预检和冻结；只有 P1、P2、P3 都达到“可直接启动正式实验”的状态后，才恢复 C/D 的 P4 修复，再完成 P4 的 E/F。即日起不继续增加 P4 机制，直到第 6 节的 P1–P3 readiness gate 全部通过。

### 当前四个子任务的证据边界

| 子任务 | 当前最好可信结论 | 不能宣称的内容 |
|---|---|---|
| P1 | 15/30 N force-step 控制在两个开发 seed 上通过，物理力/姿态记录完整 | 尚不是 full-MGA 多 seed 统计优势 |
| P2 | nominal 与 mass/friction OOD 的当前 full-MGA、no-prior、MBO 均有 2/2 strict-safe 开发结果 | 尚未完成正式 seeds 0–9；OOD 结果属于同冻结开发 probe，不能替代正式统计 |
| P3 | 同冻结 MBO 与 full-MGA 的 seed110/111 均完成 position+yaw 联合目标，physics coverage=1、physics safety violation=0；no-tangent 也为 2/2，视觉已生成；no-retraction seed110 保留真实 certified rejection | no-RL-prior 为 1/2；no-retraction seed111 两次 CPU/LLVM 编译均因内存不足、没有结果文件；reliability promotion 仍未完成 |
| P4 | 当前 no-prior 轨迹已做到峰值力 32.284 N、硬违规为零、body progress 0.204 m；旧 learned-prior 轨迹曾做到 body/support 0.326/0.267 m 和左右各一步 | 两条轨迹都 `aborted_unrecoverable` 且 task success=0；不能拼接成一次成功，也不能声称 full-MGA 已通过 |

最新检查用 GIF（均由保存的真实 q/qd 生成）位于：

- P1：`results/_development/humanoid_mga_repair/current_contract/results/humanoid/push_to_line/baseline/model_based_only/level_p1_force_15n/seed_110/trajectory/trajectory_best.gif`；
- P2：`results/_development/humanoid_mga_repair/nominal_load_retention/results/humanoid/push_to_line/baseline/model_based_only/level_p2_push_nominal/seed_110/trajectory/trajectory_best.gif`；
- P3：`results/_development/humanoid_mga_repair/nominal_load_retention/results/humanoid/push_to_line/baseline/model_based_only/level_p3_unjam/seed_111/trajectory/trajectory_best.gif`。

## 1. 当前能否做论文级 report

可以整理旧协议的全算法结果、当前 model-based 开发结果和修复诊断，但还不能报告“新 full-MGA 已在 P1–P4 全面验证通过”。旧正式结果目录已有 580 个 `results.json`；运行数量齐全不代表新任务定义、控制修复和物理指标已经完成统一验证。

当前最有价值的进展是：P1 force-step、P2 nominal 和 P3 model-based core 均有真实物理通过例；P4 分别观察到了安全推进行为和双脚支撑交换。P4 的最新 MGA 开发运行仍提前终止，而且两种能力尚未出现在同一条 full-MGA 轨迹中。

| 子任务 | 原本要验证什么 | 当前可信证据 | 距论文级报告的主要缺口 |
|---|---|---|---|
| P1：固定箱体 15/30 N 力阶跃 | 接触建立、冲击抑制、稳态力跟踪 | MBO seed110/111 的 15/30 N 均通过 force-step gate，当前 full-MGA 的 15 N 轨迹已落盘 | full-MGA/no-stiffness 的完整 15/30 N 双 seed 与正式 seeds；force-step pass 不能和 box-goal SSR 混用 |
| P2：固定站姿 10 cm 一维推线 | 在 nominal 与 mass/friction OOD 下安全推到目标 | 当前 full-MGA、no-prior、MBO 的 OOD seed110/111 均 safe success | 尚缺 nominal 的同冻结 full-MGA/no-prior/MBO 重跑与正式 seeds；OOD 仍需完整算法比较 |
| P3：平面箱体 unjamming / yaw correction | 接触几何是否帮助解卡和纠正偏航 | 同冻结 MBO seed110/111 均 position+yaw 联合成功，物理覆盖=1、physics safety violation=0；代表 GIF 已生成 | 缺 full-MGA/no-prior/no-tangent/no-retraction 的同冻结双 seed |
| P4：30–50 cm walk-and-push | 身体、支撑和箱体同步推进，同时维持接触安全 | 当前安全前缀达到 body progress 0.204 m、峰值力 32.284 N、硬违规为零；旧开发轨迹达到双脚各一步 | 当前仍在 step132 `aborted_unrecoverable`；缺单条到线+双脚步行+安全+无 abort 的 full-MGA 轨迹及两 seed 复验 |

P1/P2 不加 unjamming 墙；P3 只保留实际任务定义的障碍/接触；P4 展示真实行走推动，不能把 P1 的固定站姿视频标成 P4。P3 的几何约束与场景需明确说明，不能把有导向障碍的解卡实验称为完全无障碍自由推动。

## 2. 已观察到的改善与不能推广的调参

下面都是开发证据，不是新正式矩阵均值。

| 诊断 | 观察结果 | 对修复的含义 |
|---|---|---|
| P1 15 N，`ki_force: 1 → 0.08` | 峰值 37.820 → 35.071 N；nCVaR95 0.2768 → 0.3153；稳态 MAE 0.0108 → 0.1146 N | 较低峰值伴随更差尾部和稳态误差；仍有明显接触冲击，不能宣称所有力控指标均改善 |
| P2 nominal，同一 KI 对比 | 完成 index 36 → 39；峰值 47.534 → 49.202 N；nCVaR95 0.5219 → 0.5259 | 全局降低 KI 没有收益依据，可能更慢且力尾部更差 |
| P3 MBO seed101 | 目标误差 3.619 mm，yaw 0.01180 rad；峰值 44.086 N；335 个有效 4 ms 样本无违规 | 修复支持这一接触几何下的 model-based 可行性 |
| P3 MBO seed102 | 目标误差 4.035 mm，yaw 0.02892 rad；峰值 44.829 N；345 个有效 4 ms 样本无违规 | 另一个正例，但发生在不同源码阶段，不能合成同冻结版本的 2/2 统计 |

证据目录均位于 `results/_development/humanoid_mga_repair/integral_rate/`：

- `mbo_p3/validation.json`；
- `p3_unjam_ki0p08_seed102/validation.json`；
- `p1_force_15n_ki1p0_seed101/`、`p1_force_15n_ki0p08_seed101/`；
- `p2_push_nominal_ki1p0_seed101/`、`p2_push_nominal_ki0p08_seed101/`。

目前 P3 的依据主要是 rear-contact 有限接触范围、接触条件下的 wrench 目标与力积分处理。侧面/混合面主动选择仍需独立案例和消融，不能把当前成功例直接称为已经验证了完整 face-selection 能力。

## 3. P4 现在卡在哪里

以下文件共享后缀 `level_p4_walk_push/seed_110/results.json`，前缀均为 `results/_development/humanoid_mga_repair/`。

| 开发目录 | 预算和实际执行 | 左/右有效步数 | 箱体目标误差 | 实际物理手力峰值 | 结论 |
|---|---|---:|---:|---:|---|
| `p4_synchronized_closed_loop_dial_seed110_concrete_fix` | 完整 DIAL 开发运行 | 1 / 1 | 14.975 cm | 70.630 N | 能推进但超过 60 N，且 fall/balance violation 很高，不能作为安全成功 |
| `p4_viability3` | 请求 200；实际 137 步后拒绝；旧 prior、H8/node2 | 1 / 1 | 12.855 cm | 49.819 N | body/support progress 0.326/0.267 m、硬违规为零；证明双脚安全步态物理可达，但未到线且非当前协议 |
| `p4_support_sweep_recovery_no_prior` | 请求 200；N16、H16、node4；实际 131 步后拒绝 | 1 / 0 | 6.379 cm | 32.284 N | body progress 0.203 m、硬违规为零；首次跨过 0.20 m，但没有右脚有效步和递归后继 |
| `p4_phase_lag_rescue_provenance_no_prior` | 同一 no-prior 协议；实际 132 步后拒绝 | 1 / 0 | 6.097 cm | 32.284 N | provenance 修复使 shifted-incumbent balance risk 从约 0.048 降至 0.013，但 phase-lag rescue 只多执行一步 |

这张表只能比较诊断现象：预算、时长和组件均不同，不能作为论文公平横向比较。最新两条 MGA 开发运行没有 learned prior/reliability，所以不是 full-MGA；其力统计也只覆盖实际执行前缀，不能与完整 baseline episode 的峰值直接宣称优胜。

已确认的问题有四项：

1. **当前候选来源缺少有效的第二次支撑交换。** no-prior 路径安全推进，但在右脚有效落地前候选集合耗尽；旧 `p4_viability3` 说明 learned gait proposal 有可能补足这一方向，但旧 checkpoint/config 不能直接复用为正式证据。
2. **安全证书还不是递归可行证书。** 当前计划可以在本 horizon 内满足 force、invalid-contact/fall 与 balance gate，却可能把系统带到下一周期没有任何安全后继的状态。step131 的 phase-lag 候选当下可安全执行，step132 仍只有 balance head 超界。
3. **emergency 本身安全不等于可恢复。** task-owned UNLOAD、minimum dwell、measured phase 和 NORMAL recovery 已按实际模式预测；但从晚期边界状态卸载后仍可能无法回到双支撑可行集。需要在更早的候选选择中加入 terminal viability，而不是继续堆单帧 rescue。
4. **shift/provenance 是已修复的真实 bug，但不是唯一根因。** task-owned recovery 的 shifted incumbent 不再被普通 0.05 proposal tube 错误裁剪；风险明显下降但仍为正，说明剩余问题属于 gait/terminal viability，而非再改一个 provenance bit 即可解决。

因此不再把“增加更多 phase lag”“emergency 只要 retract 就能安全”或“单纯扩大 horizon”作为下一轮方案。下一轮必须同时解决有效 gait proposal 与 terminally viable backup；任何只把 abort 推后一两步的改动均不算通过。

## 4. P4 如何系统参考 DIAL

原始配置在 [`baselines/dial-mpc/dial_mpc/examples/unitree_h1_push_crate.yaml`](../../baselines/dial-mpc/dial_mpc/examples/unitree_h1_push_crate.yaml)：300 steps、2048 samples、H24、Hnode6、Ndiffuse4/init10、temperature 0.05；`dt=timestep=0.02`、`leg_control: torque`、目标前进速度 0.8 m/s、2 秒 ramp、`gait: slow_walk`。

对齐应分两层进行。首先核对原任务的模型/关节动作映射、reset、接触位置、步态相位、速度 ramp、奖励和 horizon，并验证原始闭环任务实际走起来；如果降低计算预算，要明确它已经不是原始配置的等价复现。然后把这一任务设定接入当前项目的 robot/task/controller 分工，在统一 impedance 与动作接口下比较 DIAL 和 MGA。

任务侧需要共同冻结：

- `gait: slow_walk`、`walk_gait_reference`、`walk_objective_mode`、速度 ramp 与 reward 的时钟含义；
- `walk_leg_control: joint_target` 与 `walk_success_mode: locomotion` 的实际执行语义；
- box mass/friction、初始间隙、手部接触、目标距离、到线停止和力 ramp；
- 物理 `timestep`、控制 `dt`、完整 episode 时间、每次候选的 dense horizon；
- 如果保留 DIAL 参考轨迹，固定生成协议、独立参考 seed、内容 hash、时间索引和终端处理，并向所有算法提供同一低层参考。

当前 reference 指向 unloaded 的开发 seed110 轨迹。可以作为修复诊断，但冻结时要明确其来源和使用方式，避免将同一个参考生成 rollout 再包装成独立验证证据。记录轨迹加残差是一个额外设计选择，并非原 DIAL 闭环 planner 的原样复刻；需要先检验它在有推力负载时是否仍适用。

严格 walking 成功建议保留当前开发判据：箱体进入目标带并满足 hold/速度条件；pelvis 前进至少 0.20 m、有效支撑前进至少 0.15 m；左右脚各至少一次物理承载的 lift–land，单步前进至少 0.04 m；全过程满足力、非手接触和姿态安全。具体参数须在两 seed 验证前固定。脚底滑动和跌倒后的位移不计作有效步行。

原始 DIAL 的 torque 接口与本项目的腿部 joint target 到力矩实现需要按代码逐项核对，不能仅凭配置名字认为不兼容。力控比较要求两算法使用相同的实际实现、物理参数和安全评估。小样本开发预算通过后，还要以拟冻结预算复验，N8 的失败不等于 N64 已失败，N8 的成功也不能替代 N64 验证。

## 5. 修复实施顺序与验收

### A. 先统一任务和报告契约

**进度：完成（代码/开发契约）。** 正式结果冻结仍属于 F，不在 A 内。

P1 的成功定义应该是力阶跃跟踪通过；P2 是 10 cm 到线；P3 同时要求位置、yaw 与解卡条件；P4 同时要求目标、真实步行和安全。rise/settling time 的计时零点和误差带在比较前固定，超时记未完成，不用任意默认时间冒充已收敛。

P1 已实现专属 force-step pass，包含预先定义的稳态误差带、保持时间、冲击限制和姿态安全；固定箱体保留的 0.5 m box goal 不再用于解释 P1 成败。P1 主报 force-step pass 与 force response，box-goal SSR 标为不适用。P1/P3 不显示目标线，P2/P4 的 marker 显示目标箱体姿态的前沿；这些只属于可视化语义，不改变内部中心坐标的 success 判据。

所有 H1 方法保存原执行的 4 ms 子步手力、非手接触、安全边界和有效采样标志；成功吸收状态之后的未执行 padding 不进入力尾部均值。记录实际完成/中止长度、首次跌倒和 safety rejection，不能将不可执行的后半段填零后称为安全。

### B. 保留 P1/P2 的有效行为，先让 P3 同版本复验

**进度：部分完成；现在的唯一 active gate。** P2-OOD 与 P3 MBO 已在当前配置上完成
双 seed；P1 full-MGA 15/30 N 和 P3 full-MGA/geometry ablation 仍未完成。在本阶段
完成前暂停新的 P4 实现与长运行。

MBO 的 P3 双 seed 已完成位置/yaw、有限 rear-contact 接触和真实物理力验收；接下来
在同一源码、同一环境定义下跑 full-MGA、no-prior 与 geometry ablation。KI 的改变
限定在有依据的 suite，不把 P3 的修复推广成所有任务默认值。

这一步验收只证明环境与 model-based 控制链可用。之后用同一环境跑 full-MGA、no-RL-prior、no-tangent、no-retraction，才能解释 proposal 和 geometry 的贡献。若 ablation 没有变差，如实报告，不为制造差距更改 baseline 定义。

### C. 统一 P4 的 gait、prior 与负载接触

**进度：接口/时钟完成，闭环未通过；当前暂停。** 已完成 DIAL reference、direct joint target、同步启动、measured support/phase 和 loaded contact 的统一；当前 canonical learned prior 尚未产生一条满足全部 P4 判据的轨迹。第 6 节通过后再恢复。

接口统一工作已经完成，下一步不再更换 gait 名称或动作映射，而是在同一当前协议下隔离 Gaussian、canonical walk prior 与 task-owned reference 的候选覆盖，确认谁真正提供第二次支撑交换。然后依次验证有负载的迈步、目标附近减速和停止。

当前 prior 的步态能力是有价值的，但还没满足“到线且力安全”。优化目标必须同时保留步态和箱体进度，避免安全卸载把正常行走长期关掉。训练继续沿用现有 H1 training script 与 curriculum，不另起训练框架。

### D. 用 111–132 步局部诊断定位 MGA 接续失败

**进度：诊断完成，递归修复未完成；当前暂停。** node/dense 一致性、mode/context、receding provenance 和 UNLOAD→NORMAL 预测路径已经逐项核对；最新明确失败点是 step132 的 balance-only terminal infeasibility。先锁定 P1–P3，再实现 terminal viability。

完整状态/任务记忆的比较已经完成：原计划 dense action、node fit、shift 后首动作、reference index、force integrator、contact acquisition、NORMAL/UNLOAD mode 与 measured anchor 均被逐项检查。node/dense 表示错误和 task-owned recovery provenance 已修复，预测与真正执行的动作现在一致。

emergency 也已按真实链条验证：先执行 minimum UNLOAD dwell，再用同一模型验证 task-owned NORMAL recovery。结果表明 late recovery 的 force 与 invalid-contact heads 可以保持安全，但 step132 的 balance terminal risk 仍为正；继续枚举 immediate roll/pitch/phase pulse 没有形成安全后继。

剩余修复不是继续定位 spline，而是在进入该边界以前要求候选具有 terminal viability：普通计划或 emergency recovery 必须结束在下一周期仍存在安全 incumbent/backup 的状态。扩大 lookahead 只能作为实现手段之一，不能把“多看一步”直接称为递归保证。

验收要求：同状态候选评估与执行一致；正常接触切换下可继续完成任务；unrecoverable rejection 如实记失败；不通过放宽力限制、允许非手推箱、减少有效步数或删除失败尾段来过关。

### E. 在物理和候选契约稳定后更新 learned components

**进度：分两段执行。** 先完成 P1–P3 fixed policy/reliability 的现行 schema 验证并纳入第 6 节冻结；P4 walk prior 与 P4 reliability 留到 C/D 修复之后。checkpoint 文件和训练入口存在，但 P4 learned components 尚未通过开发门槛。

固定站姿与 walking 的 policy/observation/action schema 分开绑定；MGA、standalone RL、ISSA 共享对应 raw policy，ATACOM 使用其匹配 tangent-action checkpoint。训练无 NaN、预算跑满只是健康检查，还需要 full-MGA 与 no-prior 的可重复结果说明 prior 有用。

H1 reliability 的 feature/risk schema、物理子步标签和候选 horizon 已变化，旧 checkpoint 不能只更换文件名继续使用。用训练/校准集重新拟合并记录 hash、样本来源及 episode 分组，在未用于拟合的开发 rollout 上检验风险错误、support abstention 和校准；相邻重叠窗口不能当成大量独立测试样本。

冻结前核对实际 `component_contract`：prior 与 learned reliability 均开启、checkpoint/schema 匹配，才标 full-MGA。若 learned confidence 全部 abstain，结果主要支持 model-based fallback，不能据此声称 learned reliability 带来改善。

### F. 两开发 seed 小矩阵 → 冻结 → 正式补跑

**进度：P1–P3 子阶段现在先做，P4 子阶段未开始。** P1–P3 可以并且应该先独立冻结到“正式可运行”；P4 仍须等 full-MGA 至少两个开发 seed 严格通过后才能写入最终 canonical 配置。

当前只执行第 6 节的 P1–P3 矩阵：P1 的 full-MGA/no-stiffness 与八算法入口检查；P2 nominal/OOD 的 MGA、no-RL-prior、MBO；P3 的 MGA、no-RL-prior、MBO、no-tangent、no-retraction，均使用两个固定开发 seed。P4 不进入这批运行。开发 seed 与训练/校准、reference 来源的重叠需要记录；已经看过的 110/111 只能作为开发验证。

晋级条件是完整任务完成、安全口径无缺失、MGA 相比 no-prior/MBO 有可解释且可重复的改进，且不靠系统性牺牲安全换进度。允许某些指标持平或某个 baseline 更好；不设“所有数字必须第一”这样的结论筛选条件。

两 seed 过关后冻结代码、canonical YAML、checkpoint、reference、指标 schema、依赖/镜像与预算。正式 seeds 仍为用户指定的 0–9，保留已接受的 overlap 声明。P4 所需 episode/horizon 可能与旧 100-step 协议不同，需在冻结前按真实 walking 所需时间确定，并在所有方法间一致。

## 6. 下一步：先把 P1、P2、P3 冻结到正式实验就绪

### B0–B5 实施状态（本轮验收记录）

本轮沿用现有 unified runner、canonical YAML 和 `results/humanoid/push_to_line`
结果协议，没有增加新的算法入口或实验框架。B0 的 H1 配置审计已通过：13 个
canonical 配置、580 个 P1–P4 运行单元、种子与 suite budget 一致，checkpoint
overlap 仅按 metadata 中的 waiver 记录。B0–B4 的配置、MGA acceptance/recovery、
task-owned rescue bank 和 report 单元测试已通过；H1 fixed-task rescue integration
也已通过。

当前可靠性 checkpoint 的 metadata 仍明确为
`performance_validated=false, promotion_eligible=false`，且它的收集 contract
覆盖旧的 45 N heavy-DR，而 canonical P2 OOD 是 35 N mass/friction。部署现在
允许该 checkpoint 仅作为 diagnostics 加载并自动 abstain；model-based physical
certificate 负责接受/拒绝，authoritative learned veto 仍会因 contract 不匹配而
拒绝。这样不会把未验证的 confidence 写成论文结论。新的 canonical full-MGA
P2-OOD 双 seed 已在这个规则下复验通过；no-RL-prior 和 model-based-only 的同一
OOD 对照也完整落盘并保持 2/2 strict-safe。B1 的 P1 15 N 已完成两个 seed
（seed110 通过，seed111 暴露一个末端 4 ms 力带反例）；P1 30 N 运行约 90 分钟
后仍无任何结果，已停止并保留该阻塞证据。B3 的 MBO/full-MGA/no-tangent 物理双
seed 已完成；no-RL-prior 为 1/2，no-retraction seed110 为真实 certified
rejection，而 seed111 两次 CPU/LLVM 编译均因内存不足退出且没有结果文件。
B4 reliability promotion 仍未完成，故当前不能宣称 P1–P3 已冻结。报告工具、
渲染器和 verifier 已在已完成的开发子矩阵上通过；它们不能把缺失的 P1/P3 物理
运行伪装成完成。

“正式实验就绪”不是要求所有 baseline 都成功，而是要求任务、full-MGA、算法绑定、指标、结果目录和运行入口已经冻结；baseline 的失败也必须是完整、可复现且可统计的真实结果。下面是恢复 P4 前必须完成的硬门槛。

### B0：冻结 P1–P3 的共同执行契约

1. 在同一个 scoped source commit 上固定 P1–P3 的 physics、controller、reward、success、安全阈值、4 ms 采样和 padding 语义。
2. 固定现有三个 suite 定义：P1 15/30 N force step；P2 nominal/mass–friction OOD 10 cm push；P3 position+yaw unjamming。P1/P2 不出现 P3 墙，P3 不显示 goal line。
3. canonical 配置只保留在 `configs/humanoid/push_to_line/`；开发 override 留在 `_development`，不增加 CPU、版本号或算法 variant 配置。
4. 固定开发 seeds110/111，只用于验收；正式 seeds 仍为 0–9。

**B0 验收：** 两个 seed 解析出的 task/config、component bindings 和输出 schema 一致；重新运行不会依赖当前 shell 的临时 override。

### B1：完成 P1 15/30 N

已通过的 MBO 轨迹作为保留基线，但还需要在同一冻结源码下完成：

1. full-MGA × 15/30 N × seeds110/111；
2. no-stiffness × 15/30 N × seeds110/111，验证 realization/stiffness 机制确实被执行；
3. 其余算法至少完成统一入口 smoke，确认固定箱体任务不会被错误的 box-goal SSR 判失败或因 action/schema 不匹配崩溃；
4. force-step pass、rise/settling、peak、overshoot、steady MAE/RMSE、nCVaR95、姿态和全部硬违规均能由统一 summarizer/report 读取；
5. 15 N 与 30 N 代表 GIF/force-time figure 从同一结果目录生成，P1 不显示绿线。

**P1 readiness：** full-MGA 两个开发 seed 的 force-step pass 均为 1、物理覆盖为 1、所有硬违规为 0；30 N 不通过降低目标力或放宽 60 N 上限过关；八算法都能由正式入口运行并产生完整结果。当前 15 N seed111 尚未满足该门槛，反例保留在开发结果目录中。

### B2：完成 P2 nominal 与 mass/friction OOD

nominal MBO 已有 2/2 safe success，下一步重点不是重复 nominal，而是解决/解释 OOD：

1. 在相同 source/config 下重跑 full-MGA、no-RL-prior、MBO × nominal/OOD × seeds110/111；
2. 定位当前 OOD 的 nonhand collision：区分真实身体/箱体碰撞、contact pair 分类错误和 OOD 参数导致的不可达状态，不能直接放宽非手接触阈值；
3. 验证 fixed-task policy 与 reliability 的 observation/action/risk schema，确保 OOD 时 learned confidence 可以拒绝错误 prior，并回退到同一 model-based certificate；
4. 保存 task/safe success、goal error、completion time、peak/nCVaR95、force/nonhand/fall/balance violation 与完整轨迹；
5. P2 绿线固定在目标箱体姿态的前沿，GIF 中最终箱体前沿应与线对齐。

**P2 readiness：** full-MGA 在 nominal 和 OOD 上均达到 seeds110/111 的 2/2 strict-safe success；MBO/no-prior 可以真实失败，但必须完整执行或以明确的 certified rejection 结束，不能因日志、schema 或 runner 错误失败。

**2026-09-16 probe 记录：** 在当前源码下 no-RL-prior 的 P2-OOD seed110 仍于
98/100 步认证拒绝，但 4 ms 物理覆盖为 1、四类硬违规为 0。第 79 步的
task-owned NORMAL rescue 已被调用；其最低 force 候选的 model-based risk 为
`[0, 0, 0, 0.993]`，说明安全 heads 已通过，之前却被通用 score-ordering 留在
UNLOAD。代码现已把 rescue 定义为“完整 NORMAL 证书通过即可接管”，并补充
geometry/load-shed 与 immediate-load 候选；这些改动已在新的 N64 CPU probe 上
复验：full-MGA、no-RL-prior 和 model-based-only 均落盘完整物理 tape，三者
P2-OOD 都是 2/2 strict-safe；因此这组结果可以计入 B2，但不能替代 P1/P3 的
冻结门槛。

### B3：完成 P3 同冻结版本的 geometry 验证

早期 seed101/102 成功例不能合并为同冻结统计。当前同冻结 MBO 的 seed110/111
已经通过；因此必须继续：

1. 保留同一冻结 source/config 的 MBO seed110/111 结果：两者均完成 position+yaw 联合目标，物理覆盖为 1 且 physics safety violation 为 0；
2. 再跑 full-MGA 与 no-RL-prior × seeds110/111；
3. 跑 no-tangent 与 no-retraction × seeds110/111，确认消融确实改变候选几何/恢复路径；不要求为了论文预设每个数都更差，但必须能解释实际差异；
4. 保存位置误差、yaw、解卡/恢复、接触面/接触点、墙接触、peak/nCVaR95、非手/跌倒/平衡和完整 abort 原因；
5. 生成成功与代表失败 GIF/俯视轨迹，P3 只显示真实 corridor/wall，不显示 goal line。

**P3 readiness：** MBO 与 full-MGA 已在同冻结版本上通过两个开发 seed；no-RL-prior
已完成并真实呈现 1/2（seed110 certified rejection、seed111 success）；no-tangent
已 2/2 通过。no-retraction 的 seed110 已真实 certified rejection；seed111 在原始
并行度和线程受限两次 CPU/LLVM 编译中均因内存不足退出，retry root 只有
`protocol_manifest.json`，没有可核验的结果文件。因此 geometry/abstention 证据
已经落盘，但 no-retraction 不能晋级为 2/2 readiness；不再引用不同 config 的
seed101/102 作为“2/2”，也不把资源阻塞改写成算法失败。

### B4：锁定 P1–P3 的 learned components 和八算法绑定

1. 检查 fixed-stance PPO、ATACOM P1/P2、ATACOM P3 与 reliability checkpoint 是否匹配当前 schema；不匹配则沿用现有训练脚本重训/重校准，不新建训练框架。
2. reliability 使用 episode 分组的独立训练/校准数据，报告 coverage、false-safe、false-reject 和校准误差；全部 abstain 只能说明 model-based fallback 工作，不能算 learned reliability 成功。
3. DIAL、MPPI、PegasusFlow、ISSA、ATACOM、standalone RL、MBO、MGA 各自保持算法级 YAML；不把无 RL 的算法包装成 MGA variants。
4. 每个方法至少在 P1、P2、P3 各完成一个正式入口 smoke，并核对实际加载的 checkpoint/hash、action width 和安全层。

**B4 验收：** unified `component_contract` 与 YAML 声明一致；没有静默 fallback、错误 checkpoint、action padding 或同名不同算法。

### B5：正式运行预检与 P1–P3 锁定

1. 用 `scripts/paper/mga/run_humanoid_push.sh` 通过统一 runner 启动 P1–P3，验证 suite 选择、resume、失败退出和结果路径；paper script 只调用现有 runner，不复制算法逻辑。
2. 对开发小矩阵运行 summarizer、renderer 和 completeness/provenance verifier，确认数据、PNG、GIF 和聚合表都能自动产出。
3. 运行 Surface Scan、PegInsert 和 H1 P1–P3 回归，形成 scoped `P1–P3 formal-ready` commit，记录 config/checkpoint/source/image hashes。
4. 此后 P4 代码只能走 P4 task-owned 接口；若必须修改共享 solver/controller，每次修改后自动重跑 P1–P3 contract tests 和最小物理回归。任何回归失败都先修复，不能带到正式矩阵。

**本轮 B5 证据：** `MGA_HUMANOID_SCOPE=p123`、`MGA_SEEDS="110 111"`
的 paper-entry dry-run 已依次解析 MGA、MBO、standalone RL、DIAL、MPPI、
PegasusFlow、ISSA、ATACOM 及 P1–P3 ablation；P2-OOD 三组开发结果的
summarizer/renderer/verifier 已通过，P3 MBO 双 seed 的结果位于
`results/_development/humanoid_p123_mbo_p3_contract`，其 verifier 为
`checked_runs=2, errors=[]`。全局 paper verifier 仍会在历史 PegInsert 配置的
缺失 checkpoint lock 处停止；这与 H1 B0–B5 结果无关，不能用错误地绑定 surface
policy 的方式掩盖。

**第 6 节总验收：** P1、P2 nominal/OOD、P3 的 full-MGA 均在同一冻结版本上通过 seeds110/111；八算法正式入口均可运行；报告/渲染/核验闭环通过。达到这一点后，P1–P3 才算“可以直接跑 seeds0–9 正式实验”，然后才能恢复 P4。

## 7. P1–P3 锁定后如何完成 P4

P4 的目标不是把 abort 再推迟几步，而是在**同一条 full-MGA 轨迹**中同时得到：目标到达和保持、左右脚真实支撑交换、body/support 推进、完整物理安全以及无 unrecoverable abort。下面按顺序执行；前一门不通过，不进入后一门。

### P4-R0：清理失败分支并建立可回退基线

1. 保留已经由反例证明有用的修复：node/dense 一致性、task-owned plan provenance、measured reference phase、UNLOAD minimum dwell、NORMAL recovery 的同模型重验和完整 abort 记录。
2. 删除或隔离只多维持一步的 phase-lag `normal_rescue_plans` 实验分支；它不是递归安全机制，不能进入 canonical 配置。保留对应失败结果，不覆盖目录。
3. 用 archived step111/124/128/131/132 状态跑定点回归，确保清理后没有恢复旧的 shift/provenance bug。
4. 跑 Surface Scan、PegInsert 与 H1 P1–P3 的受影响回归。P4 修复不得改变另外两个环境的 reward、physics、checkpoint 或落盘结果。
5. 形成 scoped commit 后再开始下一机制。后续每一阶段只新增独立开发结果目录；不使用 `v3/v4/cpu` 之类名称污染正式配置。

**R0 验收：** step111/124/128 的已修复行为不回退；P1–P3 回归通过；工作树中没有一个已知无效却默认开启的 phase-lag rescue。

### P4-R1：先证明 learned prior 提供了 Gaussian 缺少的步态方向

在完全相同的 P4 状态、物理模型、H16/node4 和安全评分下，对 archived pre-failure states 比较三组候选：Gaussian/no-prior、当前 canonical walk PPO、task-owned gait reference/incumbent。检查候选是否包含右脚 lift–land/support-exchange 所需的关节方向，而不是只比较最终 reward。

当前 `p4_viability3` 只能作为“该系统能双脚迈步”的诊断证据。它使用旧 checkpoint、H8/node2、`policy_joint_reference_residual_scale=0.20`，而当前 no-prior 路径使用 H16/node4 和更小的 proposal tube；不能直接混合指标。若 canonical walk PPO 在现行 schema 下没有产生可安全精化的右脚候选，则使用现有 `train_box_push_rl.py` 和当前 observation/action schema 重训，不新建训练框架。旧 smoke checkpoint 不晋级为正式 prior。

**R1 验收：** 在失败前状态中，learned-prior bank 至少包含一个经同一 model-based rollout 判定安全、并改善右脚支撑交换/terminal support 的候选；Gaussian bank在对应案例缺失或显著更差。若二者没有区别，如实判定 RL-prior 贡献未验证，不能靠扩大 prior 样本数制造结论。

### P4-R2：把短时安全改成 terminally viable 的递归证书

当前根因是“本 horizon 安全，但下一周期无安全后继”。因此在现有 MGA backend 和 H1 task-owned recovery 接口中加入统一的 terminal viability 检查，而不是增加更多即时 rescue：

1. 每个候选先用现有模型完成普通 horizon 的 force、invalid-contact/fall、balance 和 force-tracking 检查。
2. 在候选尾部接一个确定性的 task-owned backup：共享 gait phase/reference，加 measured roll/pitch capture，并保持相同 joint/force authority。
3. backup 必须把预测状态带回由现有物理阈值定义的可行集合：无 force/非手/fall/balance 违规、有效足底承载、躯干姿态与高度合法、下一次 shifted incumbent 或 backup 仍能通过完整证书。阈值来自冻结任务，不为某个 seed 单独修改。
4. emergency 检查完整的 `UNLOAD dwell → NORMAL gait recovery → terminal viability`，而不是只验证一帧 retract。若晚期 emergency 无法恢复，当前候选必须在更早时因 terminal infeasibility 被拒绝。
5. 对 archived step111/124/128/131/132 做离线反例测试：安全判定和真实执行使用相同 mode、task memory、force integrator、reference clock 与低层 realization。

**R2 验收：** archived 边界状态要么找到可执行且下一周期仍可行的完整链，要么在进入该状态之前被 terminal gate 提前拒绝并选择另一条有恢复余量的候选。只把 abort 从 132 移到 133 不算通过。

### P4-R3：单 seed 拉通完整 full-MGA

先用 seed110、200 steps、N16/H16/node4 做开发闭环。顺序固定为：

1. 有效 walk prior + model-based refinement/terminal gate，learned reliability 暂时关闭，用于验证物理闭环；
2. 相同配置的 no-RL-prior，验证 prior 是否确实改善候选覆盖；
3. 加载与当前风险/horizon 匹配的 learned reliability，得到真正的 full-MGA。

单 seed 必须同时满足：

- `execution_status` 不是 abort，`task_success=1` 且 `safe_success=1`；
- physics sample coverage 为 1；force、nonhand、fall、balance violation rate 全为 0；
- 实际物理手力峰值严格小于 60 N；
- body progress 不低于 0.20 m，最终 support progress 不低于 0.15 m；
- 左右脚各至少一个满足 lift–load–forward-land 定义的有效步；
- 箱体进入目标带，满足 hold 与停止条件；成功 padding 不进入物理统计。

任何一项失败都保留完整失败结果。禁止通过提高 60 N 上限、减少双脚步数、允许非手推箱、删除失败尾段或只报 prefix 指标过关。

### P4-R4：第二开发 seed、预算确认和 learned reliability

seed110 通过后，用完全相同的源码/config/checkpoint 跑 seed111。两者都通过后，再以拟冻结预算复验：MGA 使用 N64/H16/node4，P4 episode 使用 300 steps 以覆盖 DIAL 对齐的完整时间；各 baseline 保留预注册的算法级搜索参数，但共享同一 300-step 物理任务、安全阈值和指标采样。

learned reliability 必须使用当前 P4 feature、risk heads、realization 和 horizon 重新收集/校准，训练/校准 episode 与开发验收分组记录。它可以 abstain 到 model-based certificate，但不能把所有状态全部 abstain 后仍宣称 learned confidence 有效。至少报告 coverage、false-safe、false-reject、校准误差，以及 full-MGA 对 no-learned-reliability 的实际影响。

**R4 验收：** seed110/111 均完整 strict-safe success；full-MGA 相对 no-prior 至少在成功率/可恢复性或 progress–safety frontier 上给出可重复、可解释的改进；learned reliability 不引入新的 false-safe 执行。

### P4-R5：冻结、最小比较矩阵和正式运行

1. 将通过验证的字段写回现有 `configs/humanoid/push_to_line/`，不增加 CPU/GPU 或版本号配置；记录 code、reference、policy、reliability、镜像和指标 schema hash。
2. 先跑 MGA、no-RL-prior、MBO、DIAL、ATACOM × P4 × seeds110/111。MGA 未达到 2/2 时不扩大矩阵。
3. 生成真实成功 GIF、motion strip、足底承载/落脚时间条、box/body/support progress 和 6D/接触力曲线；失败 baseline 也保留完整中止或违规段。
4. 通过 completeness/provenance verifier 后，才运行正式 seeds 0–9 和剩余八算法/消融。正式统计不复用已查看的开发 seed 作为独立证据。

P4 达到论文级完成的最终定义是：冻结配置下 full-MGA 多 seed 的 strict safe success 可报告，并且与 no-prior、model-based-only 和安全 RL baseline 的差异能对应 MGA 的 prior proposal、model-based refinement、realization-aware geometry 与 receding safety 机制，而不是只展示一条挑选出的好看 GIF。

## 8. 到底需要补跑多少

旧矩阵的真实构成为：

| 范围 | 数量 |
|---|---:|
| 8 算法 × 6 suites × 10 seeds | 480 |
| no-RL-prior、no-learned-reliability × P2-OOD/P3/P4 × 10 | 60 |
| no-tangent、no-retraction × P3 × 10 | 20 |
| no-stiffness × P1 15/30 N × 10 | 20 |
| 合计 | 580 |

该 580-run 组成只描述旧结果，不再是当前正式矩阵。为与 Surface
保持同一因果消融口径，当前协议将 `no_rl_prior`、
`no_learned_reliability`、`no_controllability_geometry` 和
`no_retraction` 统一运行全部 H1 suites，并保留 P1 `no_stiffness`：

| 当前正式范围 | 数量 |
|---|---:|
| 8 算法 × 6 suites × 10 seeds | 480 |
| 4 个统一消融 × 6 suites × 10 seeds | 240 |
| no-stiffness × P1 15/30 N × 10 | 20 |
| 合计 | 740 |

`no_tangent` 保留为 P3 Appendix diagnostic，不进入 740 的正式计数。先跑
P1--P3 时，paper runner 的 `p123` scope 对应 620 runs：400 个算法运行、
200 个统一消融运行和 20 个 P1 no-stiffness 运行。

旧计划曾估算 380–580 的补跑范围，这是基于旧 policy-free 结果可能保持等价的条件。后续环境/接触控制和物理子步安全统计已经发生变化：抽查旧 MBO P3 只有 20 ms 端点轨迹，缺少原执行的 4 ms 子步信号，无法复算新口径的冲击峰值。重新 replay 可以作为新诊断，不能冒充当时执行的观测。

因此建议为当前完整协议的 740 runs 预留计算量；冻结后逐条检查哪些旧运行的动力学、reward、controller、指标采样、checkpoint 和预算完全匹配且原始数据齐全，才允许复用。若只换学习组件并证实公共环境未变，可保留对应 policy-free 行；若改变 P3/P4 任务或控制，则受影响 suite 的所有 baseline 都需重验，不能只补 MGA。

`--resume` 只有在已验证匹配冻结协议时才安全。目录里有 `results.json` 不是复用充分条件，提前中止或旧协议结果不能跳过。

## 9. 论文最终应交付什么

| 子任务 | 任务指标 | 力/安全指标 | 图与 GIF |
|---|---|---|---|
| P1 | force-step pass、rise/settling time | 真实峰值、overshoot、稳态 MAE/RMSE、违规、姿态 | 15/30 N 阶跃曲线；接触建立与稳态 GIF |
| P2 | raw/safe success、目标误差、完成时间 | 子步手力 nCVaR95/峰值、违规、非手/跌倒/平衡 | nominal/OOD 箱体轨迹+力；同协议代表 GIF |
| P3 | safe success、位置与 yaw、解卡完成 | 子步力尾部、非手/障碍接触、恢复失败 | 俯视接触/yaw 轨迹、geometry 消融；真实成功和失败 GIF |
| P4 | safe success、左右有效步数、body/support/box 位移、目标保持与停止 | 子步力尾部、姿态、非手、跌倒、emergency/rejection | 全身侧视 motion strip、落脚/承载时间条、手力；完整 walk-and-push GIF |

共同保留 per-seed reward/cost、规划耗时、candidate 总评估数与所有失败状态。additive prior 若增加额外候选，应公开额外计算量；相同 `Nsample` 不自动等于相同总 rollout 成本。

主文采用 SSR+nCVaR95 通用表，并用 P3/P4 图解释全身扩展；P1/P2 完整结果及所有 baseline/ablation 放 Appendix。报告每个 seed 和比例区间，区分 observed zero violation、模型预测的证书与绝对安全保证。

旧正式目录中的 DIAL/PegasusFlow GIF 不能证明新修复通过。当前已经从保存的真实 q/qd 生成 P1、P2、P3 的开发检查 GIF，路径列在第 0 节；它们仍是 model-based core 开发证据，不是正式 full-MGA 代表样本。P4 尚无通过全部验收的成功轨迹，因此不能生成或展示“成功 P4”论文 GIF；失败诊断媒体应保留实际中止信息。

## 10. 沿用项目架构的入口

| 工作 | 现有位置 |
|---|---|
| H1 任务、接触状态、emergency | `genedynamics/envs/domains/humanoid/box_push_brax.py` |
| 公共低层控制/机器人绑定 | `genedynamics/core/control/humanoid_contact.py`、`genedynamics/robots/h1/` |
| MGA 与 receding 逻辑 | `genedynamics/solvers/single/mga/`、`genedynamics/solvers/common/` |
| 实验适配与 suite/checkpoint 绑定 | `genedynamics/experiments/plugins/methods/contact_receding.py`、现有 framework/config |
| 指标、统计、渲染 | `genedynamics/experiments/plugins/metrics/`、`plugins/visualizations/`、`utils/` |
| 训练 | `scripts/tasks/robot/humanoid/train_box_push_rl.py`、`train_box_push_reliability.py` |
| 正式配置 | `configs/humanoid/push_to_line/` |
| 正式运行 | `scripts/paper/mga/run_humanoid_push.sh` → `python -m genedynamics.experiments.runner` |
| 报告/渲染/核验 | `scripts/paper/mga/summarize_results.sh`、`render_results.sh`、`verify_results.sh` |

当前 canonical `_base.yaml` 仍是旧 100 steps / N64 / H16 / Hnode4 与旧 P4 默认行为、旧 learned locks；开发修复不能靠临时 override 永久承载。通过验证后再将经过检验的 suite 设置及绑定写回现有配置位置。共享 solver 改动应有 Surface/PegInsert 相关回归；不改这两个任务的 reward、physics、已存结果或冻结 checkpoint。
