# Soft-Robot Co-Design — RunPod GPU 执行清单

> CPU campaign（Stages 1–7）已全部完成、43 单测全绿。本文件是上 GPU/RunPod 后**按依赖排序**的执行清单：每项含**改哪 / smoke 判据 / 喂哪张表**。所有剩余项都需要 rollout 验证，且大多共享 `mrmfmbd_mbd_jax.py` 的同一条 JIT scan，故集中在 G2 批处理。
>
> 约定：`◻ smoke` = 通过判据；env = `/opt/anaconda3/envs/fedguide` 在 GPU 机上对应的 JAX-GPU 环境。CPU 已建好的模块（`morph_system` / `estimator_system` / `mode_system` CVaR / 统一 robotize / `diffusebot_baseline` / carry）都是 JAX-native 或纯函数，GPU 上直接可用。

---

## Phase G0 — 环境 + 基线 sanity（前置，必须先过）
- [ ] **G0.1 JAX-GPU 环境**：确认 `python -c "import jax; print(jax.default_backend(), jax.devices())"` 输出 `gpu`。
- [ ] **G0.2 单测 env 平价**：跑全部 43 个新单测。
  - ◻ smoke：`pytest test/unit/test_actuator_weight_stress.py test/unit/test_morph_decoder.py test/unit/test_cv_estimator.py test/unit/test_cvar_regime.py test/unit/test_unified_robotize.py test/unit/test_prior_adapters.py test/unit/test_carry_reward.py test/unit/test_diffusebot_baseline.py` → **48 passed / 2 skip**。
- [ ] **G0.3 报告基线复跑**：`run_co_design_with_gif.py configs/soft_robot/main/crawling_ground.yaml`（report 的结果）。
  - ◻ smoke：`results.json` 生成、final reward > 0、gif 出图。**这一步先于任何新 wiring**，确认 sim+MBD 在 GPU 通。

---

## Phase G1 — Stage 1 全 rollout 等价性（地基，零新行为，只验证）
- [ ] **G1.1 one-hot 全 rollout == legacy**：构造 `SoftBodySpec` 带 `actuator_weight = one-hot(actuator_id)`（passive→零行），`rollout_return` 跑 with vs without。
  - ◻ smoke：对 3–5 组随机 (x,φ)，`max|Δreward| < 1e-4`。kernel 级已证（diff 2e-11），这步是端到端确认，**解锁 A2/DiffuseBot 的加权 actuation 路径**。
  - 喂：贡献1 设计空间对齐的正确性。

---

## Phase G2 — Hot-loop wiring（核心；共享 `mrmfmbd_mbd_jax.py` 的 `_make_block_runner`/`step`）
> 顺序重要：四个子项都改同一条 scan。建议按 a→b→c→d，每个 wiring 后立刻 smoke 再叠下一个。所有新逻辑都已在 CPU 子包里写好+测过，这里是**接线 + JAX 镜像 + rollout 验证**。

- [ ] **G2a A2 decoder 插 rollout 前**（Stage 2 收尾）
  - 改：`MBDConfig` 加 `morph_latent_dim` / `morph_decoder_path` 旋钮；`mrmfmbd_baseline.py` 用 `MorphDecoder.load(path, cfg)` 构造、传入 backend；`_rollout_and_marginalize` 里 x-block 维度=latent，rollout 前 `x_morph = decoder.decode(w)`（已 jittable）。
  - ◻ smoke：tiny（K5,M4,C1）+ `morph_latent_dim=8` + `data/morph_decoders/loco_cpu` → 无报错；x-block 维度=8；decode 出的 occupancy ∈ [x_lo,x_hi]；reward 有限。
  - 喂：贡献1 headline joint MF/MB、Table 1 Ours、Table 3。
- [ ] **G2b prior log_prob 折进权重**（Stage 2 收尾）
  - 改：w-block prior 设 N(0,I)（latent 的 `x_mean=0,x_std=1`）；`_weighted_mean` 的 `log_w` 加 `theta_prior.log_prob_batch`（已算好、当前没用）；旋钮 `use_prior_weight`。
  - ◻ smoke：开 `use_prior_weight` → 权重偏向低 ‖w‖ 候选；ESS 合理；tiny reward 不比关时差。
- [ ] **G2c CV 估计器 + 解非递减 + dual-ν**（Stage 3 收尾，核心 ML）
  - 改：把 `estimator_system/control_variate.py` 的 numpy 逻辑 JAX 镜像进 scan（low-fi 全 M + hi-fi 子集校正，替换"选一层 ℓ"）；移除/放宽 `mbd_jax.py:336` 的非递减 `raise`；接 `BudgetDual`（`nu_max` 旋钮）。
  - ◻ smoke：tiny + `method=control_variate` → 跑通；每步用到两层 fidelity；记录的 score 方差 < 单保真度（matched budget）；fidelity 可非单调。
  - 喂：**贡献2（核心）**、Table 4、Fig fidelity。
- [ ] **G2d CVaR 接 marginalizer 分派**（Stage 4 收尾）
  - 改：`mbd_jax.py` 的 marginalizer 分派加 `regime_posterior_mode=="cvar"` → `cvar_marginalize_jax`（已写）；`MBDConfig` 加 `cvar_alpha` 旋钮（类比 `risk_temperature`）。
  - ◻ smoke：tiny + `regime_posterior_mode=cvar, cvar_alpha=0.3` → 跑通；worst-mode return 优于 `reward` 模式；q 集中低-reward regime。
  - 喂：贡献4、Fig regime、Table 1 robustness 列。

---

## Phase G3 — Baseline + 任务 smoke
- [ ] **G3.1 DiffuseBot 梯度 smoke**：`diffusebot` baseline tiny config。
  - ◻ smoke：梯度不 NaN（actor_grad_norm 有限，参考 README §5 debug 表：h 短、lr 小、grad_clip）；reward 随 iter 上升。
  - 喂：贡献1（gradient-free vs gradient-based）、Table 1 "DiffuseBot-style" / "first-order ours"。
- [ ] **G3.2 Carry 任务接线**：evaluator 的 task_id 分派加 `carry` → `rollout_return_carry`；写 `configs/soft_robot/main_v2/carry.yaml`（mirror `push.yaml`，`manip_cfg.horizontal_only=False`）+ regime bank（object 质量/坡度）。
  - ◻ smoke：carry rollout 跑通；无支撑/无凹腔的体 → 物体掉落（`G_T` 低）；好体 → 托运（`G_T` 高、object forward disp 正）。
  - 喂：loco-manip 表、贡献1/3/4 showcase。

---

## Phase G4 — Asset banks（关 Gap A）
- [ ] **G4.1 clone 3D priors**：`third_party/morphology_priors/triposg`（主）→ `git clone ... && pip install -r requirements.txt`；可选 hunyuan3d/trellis/craftsman。验证 `get_prior("triposg").metadata`。
- [ ] **G4.2 每 prior 建 bank**：`build_asset_bank.py --prior <p>` + `robotize_bank.py`。
  - ◻ smoke：每 prior RobotizationSuccess ≥ 30%（random_shapes 100%，triposg 预期 50–70%）；manifest + robotized/*.npz 生成。
- [ ] **G4.3 （可选）重训 decoder**：`train_morph_ae.py --bank-root <triposg bank>`（比 loco_cpu 更丰富的先验）→ 新 `data/morph_decoders/<prior>`。
  - 喂：Table 3 prior 横评、更强的 A2 先验。

---

## Phase G5 — 渲染器（关 Gap B）
- [ ] **G5.1 写 `scripts/visualizations/render_soft_robot_checkpoints.py`**（PyVista 起步）：输入 `results/.../results_seed_*.json` 的 `theta_history`，对选定 checkpoint re-rollout `(x_k,φ_k)` 取粒子位置，按 `actuator_id` 着色，5-列演化网格。
  - ◻ smoke：产出 `figures/evolution.png`（5 列）；re-rollout 不报错。
  - 喂：Fig evolution（DiffuseBot 风格）。

---

## Phase G6 — Phase 5 sweeps + 出表出图（大算力，~300–500 GPU-h）
- [ ] **G6.1 sweep runner**：`scripts/sweeps/run_sweep.py`（README §3.2 草稿）支持 per-seed override。
- [ ] **G6.2 Table 1** 主表：11 methods × 3 tasks × 5 seeds（含 diffusebot/first-order、Ours full、Ours+SHAC）。
- [ ] **G6.3 Table 2** controller-only / **Table 3** prior 横评 / **Table 4** fidelity ablation / **Fig regime** / **loco-manip 表** / **(可选 appendix) generality**（估计器搬 `double_integrator_box`/`drone_box_3d`）。
- [ ] **G6.4 出表出图**：`make_table1.py`/`make_table4.py`/`make_figure3.py` + 现有 `analyze_*`/`compare_*` 聚合多 seed。
  - 喂：全部 paper Tables/Figures。

---

## 依赖图（一句话）
G0 → G1 → **G2(a→b→c→d)** 是关键链（hot-loop）；G3 可在 G1 后并行（baseline/carry 不依赖 G2 的 wiring）；G4/G5 独立（离线/渲染）；G6 依赖 G2+G3+G4+G5 全绿。

## 一次性最小可发布路径（若想先出一版结果）
G0 → G1 → G2a+G2b（joint prior 跑通）→ G2c（核心 ML）→ G3.1（DiffuseBot 对照）→ G4.2（triposg bank）→ G6.2 缩减版 Table 1（locomotion 单任务）。其余（CVaR/carry/generality/全 sweep）作后续补全。

---

# 附录 A — 如何跑 benchmark / baseline

## A.1 单次实验（一个 method × 一个 task × seeds）
入口统一是 `run_co_design.py`（或带渲染的 `run_co_design_with_gif.py`），吃一个 YAML：
```bash
python scripts/tasks/soft_robot/co_design/main/run_co_design.py <config.yaml>
# → results/soft_robot/.../<exp>/results_seed_*.json
```
YAML 关键字段（决定跑哪个 method / task / 指标）：
```yaml
baseline_name: mrmfmbd        # ← 选 baseline（注册名，见 A.2）
task_id: crawling_ground       # ← 选任务：crawling_ground / locomotion / push / carry
seeds: [0,1,2,3,4]
evaluator_runtime:
  voxel_dims: [4,3,4]          # 形态栅格（mesh 路 3,3,3）
  reward_shaping_weight: 100.0
  # regime_bank_kind / softbody_spec_path / morph_decoder_path 等按 stage 加
method_params:                 # ← 透传给 solver（BaselineConfig.extra）
  num_modes: 4                 # regime 数（指标按这些 regime 聚合）
  num_fidelity_levels: 3
  regime_posterior_mode: risk_sensitive   # reward / risk_sensitive / cvar
  # A2: morph_latent_dim, morph_decoder_path
  # G2c: method=control_variate, nu_max
  # G2d: cvar_alpha
```
不改代码可调的旋钮见 `configs/soft_robot/README.md` §"Knobs you toggle WITHOUT touching code"。

## A.2 Baseline 清单（注册名 → 怎么调）
`baseline_name:` 从 registry 取（`list_baselines()` = `mrmfmbd / cmaes / cem / shac / diffusebot`）。Table 1 的 11 行：

| # | Method | baseline_name + config | 说明 |
|---|---|---|---|
| 1 | Random prior | `cem`，1 代 popsize=1 | 无优化下界 |
| 2 | CEM | `cem` — `baselines/cem_crawling*.yaml` | 黑盒 |
| 3 | CMA-ES | `cmaes` — `baselines/cmaes_crawling*.yaml` | 黑盒 |
| 4 | MBD (vanilla) | `mrmfmbd` — `main/crawling_ground.yaml` | 单 mode 单 fidelity |
| 5 | MBD + regime | `mrmfmbd` + `regime_posterior_mode: risk_sensitive`（或 `cvar`） | 贡献4 |
| 6 | MBD + adaptive iALM | `mrmfmbd` + `nu_max>0`（`crawling_alm.yaml`） | 预算 dual-ν |
| 7 | **DiffuseBot-style** | `diffusebot` | 梯度 co-design（Stage 5） |
| 8 | SHAC-only | `shac` — `shac_only_crawling.yaml` | 梯度、固定形态 |
| 9 | **Ours (full)** | `mrmfmbd` + A2 prior + `method=control_variate` + risk/cvar + nu_max | 全套 |
| 10 | Ours + SHAC refine | `crawling_alm_shac.yaml` | hybrid 0阶+1阶 |
| 11 | first-order ours (ablation) | `diffusebot` + `prior_guidance_weight>0` | 把 MC score 换成 ∇R |

**公平性**：所有 method 同一 evaluator、同一 `num_modes` 的 regime 集、同一总预算（`K·M·num_modes` rollouts；CEM/CMA-ES 用 `popsize×generations` 匹配）。gradient-free（mrmfmbd）vs gradient-based（diffusebot/shac）的对照就是贡献1 的实锤。

## A.3 多 seed sweep + 聚合
```bash
python scripts/sweeps/run_sweep.py <config1> <config2> ... --seeds 0,1,2,3,4   # G6.1
python scripts/tasks/soft_robot/co_design/analysis/analyze_main_result.py <exp>  # mean/std across seeds
python .../compare_4_methods.py / compare_fidelity_ablation.py / compare_reward_curves.py
```
每个 run 写 `results_seed_*.json`（含 `theta`、`x`、`phi`、per-regime returns、wall、`theta_history`）。聚合脚本算 mean±std、出曲线。

---

# 附录 B — 指标（定义 / 在哪算 / 喂哪个问题）

记号：morphology–controller 对 `(x,φ)`，regime `m∈M`（如 ground friction μ∈{0.3,0.4,0.5,0.6}），高保真 reward `R(x,φ;m)` = 200 步后质心前向位移。evaluator 返回 per-(candidate, regime) returns；`results_seed_*.json` 存下，聚合脚本算下列量。

## B.1 主指标（Q1 质量 / Q2 鲁棒）
| 指标 | 定义 | 喂 |
|---|---|---|
| **¯R**（in-dist mean） | `mean_{m∈M} R(x⋆,φ⋆;m)` | Q1、Table 1 主列 |
| **per-regime return** | 每个 μ 下的 `R(x⋆,φ⋆;μ)`（含 held-out） | 跨 regime 表/热图 |
| **R_min**（worst-mode） | `min_{m∈M} R(x⋆,φ⋆;m)` | **Q2 鲁棒（贡献4 主指标）** |
| **min/max ratio** | `R_min / max_m R` | regime profile 是否被压平 |
| **wall-clock (s)** | 端到端墙钟 | 成本 |

## B.2 多保真指标（Q3 / 贡献2 核心）
| 指标 | 定义 | 喂 |
|---|---|---|
| **return vs cumulative substeps** | 运行最优 return 对累计物理 substep（= 真实计算量）作曲线 | **Pareto（Fig fidelity）** |
| **% of high-fid return @ fraction cost** | adaptive 在 X% wall-clock 下达到 fixed-`|m|=200` 的百分比 | Table 4（如"89% return @ 67% wall"） |
| **score-estimate variance @ matched budget** | 固定预算下 score（加权均值）方差，CV vs 单保真 | 贡献2 估计器（CPU 已证、GPU 复现） |
| **dual-ν trace / 选中的 ℓ_k** | 每 denoising step 的 ν 与所选 fidelity | 自适应行为（非单调）佐证 |
| **ESS** | `1/Σw²`，重要性权重有效样本数 | 采样健康度 |

## B.3 Prior 横评（Table 3 / 贡献3）
每 prior 量四项（`robotize_bank` + 下游 co-design）：
| 指标 | 定义 |
|---|---|
| **RobotizationSuccess %** | 通过 validity 门（连通/接地/actuator 覆盖）的资产占比（target ≥30%） |
| **diversity** | 资产 occupancy 的两两距离 / 覆盖度（避免模式坍缩） |
| **downstream best ¯R** | 该 prior 的 bank 上 co-design 得到的最优 ¯R |
| **sim stability** | rollout NaN / 爆炸率（生成体的可仿真性） |

## B.4 Loco-manipulation（carry/push）
| 指标 | 定义（来自 `carry_reward` / push reward） |
|---|---|
| **object Δx** | 物体前向位移 |
| **G_T（carried fraction）** | `mean_t 1[obj_y ≥ carry_y_min]` —— 接触稳定（carry 专属） |
| **dist-to-goal_T** | 终态物体到目标 x 的距离 |
| 跨 object regime | 上述指标在 object 质量/摩擦/尺寸 regime 上的 ¯· 与 worst |

## B.5 报表口径
- 每 cell = `mean ± std` over 5 seeds；主表对每 method × task 报 ¯R / R_min / wall。
- 曲线：return-vs-iteration（收敛）+ return-vs-substeps（成本）。
- 显著性：方法间用 5-seed 的 paired 比较（AAAI/ICLR 期望多 seed 统计）。
- 消融：Ours 去 A2 prior / 去 CV（单保真）/ 去 regime（reward 模式）/ 去 dual-ν，各自对应 ¯R / R_min / substeps 的退化。
