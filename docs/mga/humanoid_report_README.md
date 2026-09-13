# H1 Push-to-Line：P1–P4 论文报告缺口与修复顺序

审计日期：2026-09-12。本文依据落盘结果、当前代码/config 和 [experiment_run_plan.md](experiment_run_plan.md) 整理状态。这里的 P1–P4 是四类 H1 子任务，与工程实施计划 P0–P12 的编号不同。

三页图表布局见 [paper_figures_README.md](paper_figures_README.md)。本次只整理文档，没有新训练、新仿真或新的成功 GIF。

## 1. 当前能否做论文级 report

可以整理旧协议的全算法结果和修复诊断，但还不能报告“新 full-MGA 已在 P1–P4 全面验证通过”。旧正式结果目录已有 580 个 `results.json`；运行数量齐全不代表新任务定义、控制修复和物理指标已经完成统一验证。

当前最有价值的进展是：P3 的 model-based 控制出现了安全完成的开发例；P4 的 standalone prior 已经能真实迈步。但 P3 还没有同一冻结版本的 full-MGA 配对验证，P4 的 MGA 开发运行仍提前终止。

| 子任务 | 原本要验证什么 | 当前可信证据 | 距论文级报告的主要缺口 |
|---|---|---|---|
| P1：固定箱体 15/30 N 力阶跃 | 接触建立、冲击抑制、稳态力跟踪 | 新物理统计下有 15 N、MBO seed101 配对诊断，无观察到的力上限/非手/平衡/跌倒违规 | 30 N、新版本多 seed、full-MGA 和 no-stiffness；明确 force-step 专用通过条件，不能用箱体到线 SSR |
| P2：固定站姿 10 cm 一维推线 | 在 nominal 与 mass/friction OOD 下安全推到目标 | 新源码下 nominal seed101 的两种积分增益均 safe success | 同一版本 nominal/OOD 多 seed；有用的 prior、匹配的 reliability，以及完整算法比较 |
| P3：平面箱体 unjamming / yaw correction | 接触几何是否帮助解卡和纠正偏航 | MBO seed101/102 各有 position+yaw 联合成功和 4 ms 子步安全记录 | 两例源码不同；尚非 full-MGA；缺同版本两 seed 与 no-prior/no-tangent/no-retraction 对比 |
| P4：30–50 cm walk-and-push | 身体、支撑和箱体同步推进，同时维持接触安全 | unloaded DIAL reference 有步态；standalone prior seed110 左右脚各 2 步、无跌倒 | MGA 在第 114 步附近被拒绝；尚未统一 slow_walk 与训练/执行契约；缺到线+真实步行+安全的两 seed 完整验证 |

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
| `p4_dial_loaded_slow_walk_reference` | 180 steps；N64、H24 | 0 / 0 | 3.808 cm | 98.116 N | SSR=0，有滑脚及力/非手/平衡违规 |
| `p4_standalone_prior_isolation` | 完整 160 steps | 2 / 2 | 4.371 cm | 89.624 N | SSR=0，无跌倒/非手/平衡违规，但有力超限且未进目标带 |
| `p4_mga_additive_closed_loop` | 请求 160；N8、H8、Hnode2；实际 114 步后拒绝 | 1 / 1 | 10.461 cm | 37.205 N | 实际前缀无违规；`aborted_unrecoverable`，未成功 |
| `p4_mga_additive_recursive_backup` | 同上；加入第二步 backup 检查 | 1 / 1 | 10.464 cm | 37.205 N | 仍在 114 步附近拒绝，未解决接续失败 |

这张表只能比较诊断现象：预算、时长和组件均不同，不能作为论文公平横向比较。两个 MGA 开发运行的 `component_contract.learned_reliability=false`，所以它们也不是完整 full-MGA。37.205 N 属于 `partial_metrics`，不能拿短前缀的最大力对比完整 standalone 轨迹的最大力，宣称 MGA 已胜出。

已确认的问题有三项：

1. **步态契约仍未统一。** 最近 standalone 和两次 additive 的 `config_snapshot.env_params` 没有 `gait` 字段，当前 `HumanoidBoxPushConfig.gait` 默认是 `jog`；上述 DIAL 对照则显式 `slow_walk`。因此不能说这几次运行已经完成 slow_walk 同配置验证。
2. **安全动作的接续没有通过。** 最后几步观察到：一次 UNLOAD 可执行，下一次 incumbent/refined/emergency 均预测不安全，运行被拒绝。当前普通 P4 候选检查前两个 interval，`score_emergency` 只检查一个 UNLOAD interval。短时通过不足以证明下一步仍有安全可行候选。
3. **低风险选择同时损失了进度。** prior 能迈步，refinement/gate 后只走到更短的安全前缀。需要定位是候选覆盖不足、评价和实际执行不一致，还是状态已经接近可恢复边界，而不能持续增大安全惩罚期待自然修好。

“spline shift 有 bug”“emergency 只要 retract 就能安全”“扩大 horizon 就会成功”目前都是待检验的解释。现有结果未确认唯一根因。

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

P1 的成功定义应该是力阶跃跟踪通过；P2 是 10 cm 到线；P3 同时要求位置、yaw 与解卡条件；P4 同时要求目标、真实步行和安全。rise/settling time 的计时零点和误差带在比较前固定，超时记未完成，不用任意默认时间冒充已收敛。

当前 P1 尚未实现专属 force-step pass：固定箱体仍保留 0.5 m box goal，`_task_reached` 与通用 `safe_success` 仍依赖到线条件。因此 P1 的 SSR=0 不能解释为力控失败。应在现有任务指标层补充 force-step pass，包含事先规定的稳态误差带、保持时间、冲击限制和姿态安全。在此契约完成前，P1 主报已有的 force response 指标，box-goal SSR 标为不适用。

所有 H1 方法保存原执行的 4 ms 子步手力、非手接触、安全边界和有效采样标志；成功吸收状态之后的未执行 padding 不进入力尾部均值。记录实际完成/中止长度、首次跌倒和 safety rejection，不能将不可执行的后半段填零后称为安全。

### B. 保留 P1/P2 的有效行为，先让 P3 同版本复验

在同一源码、同一环境定义下重测 P3 MBO 两个开发 seed，确认位置/yaw、有限 rear-contact 接触和真实物理力都通过；同时用 P1 15/30 N、P2 nominal/OOD 做保留检查。KI 的改变限定在有依据的 suite，不把 P3 的修复推广成所有任务默认值。

这一步验收只证明环境与 model-based 控制链可用。之后用同一环境跑 full-MGA、no-RL-prior、no-tangent、no-retraction，才能解释 proposal 和 geometry 的贡献。若 ablation 没有变差，如实报告，不为制造差距更改 baseline 定义。

### C. 统一 P4 的 gait、prior 与负载接触

先统一实际使用的 slow_walk/时钟/观测/action mapping，再依次验证 unloaded locomotion → 有负载的接触与迈步 → 目标附近减速和停止。重验 standalone prior 的真实步态，明确其力峰值出现于接触建立、支撑切换还是靠近目标阶段。

当前 prior 的步态能力是有价值的，但还没满足“到线且力安全”。优化目标必须同时保留步态和箱体进度，避免安全卸载把正常行走长期关掉。训练继续沿用现有 H1 training script 与 curriculum，不另起训练框架。

### D. 用 112–114 步局部诊断定位 MGA 接续失败

使用完整状态/任务记忆，比较以下对象：原计划预测的第二个 dense action、真正 shift 后的首个 dense action、下一周期实际评分与执行的动作。同步检查 reference index、force integrator、contact acquisition、NORMAL/UNLOAD mode 以及 measured-anchor 更新。

若预测第二步与 shift 后动作不同，先区分 spline 插值/重拟合误差与其他映射变化，再在已有 spline/receding 实现中修复；若动作相同但状态不同，检查 rollout–execution 的低层控制、任务记忆和模式交接。只有定位之后才改代码，回归测试应覆盖这一具体失败。

同时检验 emergency 后继状态是否存在可继续执行的安全动作。正常两步证书不能自动覆盖仅评估一步的 UNLOAD。需要在 task-owned emergency 中定义卸载后的支撑/关节行为与可恢复条件，统一其预测和执行；验证失败则提前选择更有恢复余量的候选。扩大 lookahead 可以作为诊断，但“多看一步”本身不构成递归可行性证明。

验收要求：同状态候选评估与执行一致；正常接触切换下可继续完成任务；unrecoverable rejection 如实记失败；不通过放宽力限制、允许非手推箱、减少有效步数或删除失败尾段来过关。

### E. 在物理和候选契约稳定后更新 learned components

固定站姿与 walking 的 policy/observation/action schema 分开绑定；MGA、standalone RL、ISSA 共享对应 raw policy，ATACOM 使用其匹配 tangent-action checkpoint。训练无 NaN、预算跑满只是健康检查，还需要 full-MGA 与 no-prior 的可重复结果说明 prior 有用。

H1 reliability 的 feature/risk schema、物理子步标签和候选 horizon 已变化，旧 checkpoint 不能只更换文件名继续使用。用训练/校准集重新拟合并记录 hash、样本来源及 episode 分组，在未用于拟合的开发 rollout 上检验风险错误、support abstention 和校准；相邻重叠窗口不能当成大量独立测试样本。

冻结前核对实际 `component_contract`：prior 与 learned reliability 均开启、checkpoint/schema 匹配，才标 full-MGA。若 learned confidence 全部 abstain，结果主要支持 model-based fallback，不能据此声称 learned reliability 带来改善。

### F. 两开发 seed 小矩阵 → 冻结 → 正式补跑

优先矩阵：MGA、no-RL-prior、MBO、DIAL × P2-OOD/P3/P4 × 两个固定开发 seed，共 24 runs；外加 P1 15/30 N 的回归。P3 geometry 消融先在两 seed 做针对性检查，再跑正式 seeds。开发 seed 与训练/校准、reference 来源的重叠需要记录；已经看过的 110/111 只能作为开发验证。

晋级条件是完整任务完成、安全口径无缺失、MGA 相比 no-prior/MBO 有可解释且可重复的改进，且不靠系统性牺牲安全换进度。允许某些指标持平或某个 baseline 更好；不设“所有数字必须第一”这样的结论筛选条件。

两 seed 过关后冻结代码、canonical YAML、checkpoint、reference、指标 schema、依赖/镜像与预算。正式 seeds 仍为用户指定的 0–9，保留已接受的 overlap 声明。P4 所需 episode/horizon 可能与旧 100-step 协议不同，需在冻结前按真实 walking 所需时间确定，并在所有方法间一致。

## 6. 到底需要补跑多少

旧矩阵的真实构成为：

| 范围 | 数量 |
|---|---:|
| 8 算法 × 6 suites × 10 seeds | 480 |
| no-RL-prior、no-learned-reliability × P2-OOD/P3/P4 × 10 | 60 |
| no-tangent、no-retraction × P3 × 10 | 20 |
| no-stiffness × P1 15/30 N × 10 | 20 |
| 合计 | 580 |

不应现在就重跑 580。先完成上述局部修复和两 seed 门槛，减少无效大矩阵。

旧计划曾估算 380–580 的补跑范围，这是基于旧 policy-free 结果可能保持等价的条件。后续环境/接触控制和物理子步安全统计已经发生变化：抽查旧 MBO P3 只有 20 ms 端点轨迹，缺少原执行的 4 ms 子步信号，无法复算新口径的冲击峰值。重新 replay 可以作为新诊断，不能冒充当时执行的观测。

因此建议为完整 580 预留计算量；冻结后逐条检查哪些旧运行的动力学、reward、controller、指标采样、checkpoint 和预算完全匹配且原始数据齐全，才允许复用。若只换学习组件并证实公共环境未变，可保留对应 policy-free 行；若改变 P3/P4 任务或控制，则受影响 suite 的所有 baseline 都需重验，不能只补 MGA。

`--resume` 只有在已验证匹配冻结协议时才安全。目录里有 `results.json` 不是复用充分条件，提前中止或旧协议结果不能跳过。

## 7. 论文最终应交付什么

| 子任务 | 任务指标 | 力/安全指标 | 图与 GIF |
|---|---|---|---|
| P1 | force-step pass（定义后）、rise/settling time | 真实峰值、overshoot、稳态 MAE/RMSE、违规、姿态 | 15/30 N 阶跃曲线；接触建立与稳态 GIF |
| P2 | raw/safe success、目标误差、完成时间 | 子步手力 nCVaR95/峰值、违规、非手/跌倒/平衡 | nominal/OOD 箱体轨迹+力；同协议代表 GIF |
| P3 | safe success、位置与 yaw、解卡完成 | 子步力尾部、非手/障碍接触、恢复失败 | 俯视接触/yaw 轨迹、geometry 消融；真实成功和失败 GIF |
| P4 | safe success、左右有效步数、body/support/box 位移、目标保持与停止 | 子步力尾部、姿态、非手、跌倒、emergency/rejection | 全身侧视 motion strip、落脚/承载时间条、手力；完整 walk-and-push GIF |

共同保留 per-seed reward/cost、规划耗时、candidate 总评估数与所有失败状态。additive prior 若增加额外候选，应公开额外计算量；相同 `Nsample` 不自动等于相同总 rollout 成本。

主文采用 SSR+nCVaR95 通用表，并用 P3/P4 图解释全身扩展；P1/P2 完整结果及所有 baseline/ablation 放 Appendix。报告每个 seed 和比例区间，区分 observed zero violation、模型预测的证书与绝对安全保证。

当前旧正式目录仅查到 7 个 GIF：DIAL 的旧 P4 seed0，以及 PegasusFlow 的 6 个 suites seed0。它们不能证明新修复通过；上述新 P3 成功目录尚未发现 GIF，最新 P4 没有通过验收的成功轨迹可供出图。后续应从已达标、同冻结协议的执行轨迹生成成功 GIF，失败诊断媒体也保留实际中止信息。

## 8. 沿用项目架构的入口

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
