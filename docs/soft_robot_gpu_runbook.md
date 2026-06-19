# Soft-Robot Co-Design — RunPod GPU 执行清单

> CPU campaign（Stages 1–7）已全部完成、43 单测全绿。本文件是上 GPU/RunPod 后**按依赖排序**的执行清单：每项含**改哪 / smoke 判据 / 喂哪张表**。所有剩余项都需要 rollout 验证，且大多共享 `mrmfmbd_mbd_jax.py` 的同一条 JIT scan，故集中在 G2 批处理。
>
> 约定：`◻ smoke` = 通过判据；env = `/opt/anaconda3/envs/fedguide` 在 GPU 机上对应的 JAX-GPU 环境。CPU 已建好的模块（`morph_system` / `estimator_system` / `mode_system` CVaR / 统一 robotize / `diffusebot_baseline` / carry）都是 JAX-native 或纯函数，GPU 上直接可用。

> ### 🆕 架构 & prior 升级（2026-06-18，已落地+回归）
> **① baseline→solver 重构（用户拍板，已完成）**：`experiments/framework/baselines/` 目录 + `baseline_registry`/`BaselineProtocol` dispatch **已删除**。experiment 是通用框架,通过 **co-design solver registry** 按名跑注册的 solver。co-design 表达成 horizon-1 (state,action) 问题:`envs/external/jax_mpm/codesign_env.py` 的 `SoftRobotCoDesignDynamics(DynamicsModel)`(action=零中心 a,`θ=mean+scale·a`,`jax_transition`=robotize+MPM→reward) + `CoDesignEnergy`=-reward。`cem` = **通用 `CEMSolver` + 这个 env 桥**(通用算法零改动);mrmfmbd/cmaes/shac/diffusebot 挪到 `solvers/single/codesign_optimizers/`,由 `solvers/single/codesign_solvers.py` 注册(G2 机制 bit-identical 保留)。runner = `experiments/framework/codesign_runner.run_codesign`,platform dispatch 改 `get_codesign_solver`。**回归:gpusmoke per-seed 全字段保留 + 43 单测全绿。** follow-on:cmaes 改 general CMAESSolver+env;MRMFMBD 深度归一到 (dynamics,energy,prior)。
> **② 3D prior 候选锁定（text 主轴 + image/A2 可插）**:见 Phase G4 + paper plan。text 线:Point-E / Shap-E / TRELLIS-text / **TRELLIS.2-O-Voxel**(CVPR'26,2512.14692) / **DiffGS**(3DGS latent diffusion,A2 basis 优选) / **Turbo3D**(text→3DGS) / SplatFlow / UVGS。image/mesh/A2-latent 臂:**MeshFlow**(CVPR'26,MeshVAE 紧凑 latent = A2 优选,**非 text**) + TripoSG(G4 已通)。4DGS 不做;3DGS 空壳→`gs_robotize` 要填实。**联网只主循环能上,subagent 断网。**

---

## Phase G0 — 环境 + 基线 sanity（前置，必须先过）
- [x] **G0.1 JAX-GPU 环境**：✅ JAX 0.6.2，`gpu` / `CudaDevice(id=0)`（RTX A5000）。注：pip 环境需先 `pip uninstall pin`（pinocchio 段错误）+ `pip install scikit-image`，见 plan 文档。
- [x] **G0.2 单测 env 平价**：✅ 43 passed（修完 pinocchio/skimage 后）。跑全部 43 个新单测。
  - ◻ smoke：`pytest test/unit/test_actuator_weight_stress.py test/unit/test_morph_decoder.py test/unit/test_cv_estimator.py test/unit/test_cvar_regime.py test/unit/test_unified_robotize.py test/unit/test_prior_adapters.py test/unit/test_carry_reward.py test/unit/test_diffusebot_baseline.py` → **48 passed / 2 skip**。
- [x] **G0.3 报告基线复跑**：✅ return_=11.78、success、36 min/seed、gif 出图。`run_co_design_with_gif.py configs/soft_robot/co_design/main/crawling_ground.yaml`（report 的结果）。注：`data/{asset_banks,morph_decoders}/loco_cpu` 原缺失，已用 `build_asset_bank→robotize_bank→train_morph_ae`（random_shapes，3,3,3，latent 8）重建（decoder MSE 8e-4）。
  - ◻ smoke：`results.json` 生成、final reward > 0、gif 出图。**这一步先于任何新 wiring**，确认 sim+MBD 在 GPU 通。

---

## Phase G1 — Stage 1 全 rollout 等价性（地基，零新行为，只验证）
- [x] **G1.1 one-hot 全 rollout == legacy**：✅ max|Δreward|=5.9e-5 < 1e-4（5 组随机 x,φ；脚本 `scripts/.../co_design/smoke/g1_rollout_equivalence.py`）。构造 `SoftBodySpec` 带 `actuator_weight = one-hot(actuator_id)`（passive→零行），`rollout_return` 跑 with vs without。
  - ◻ smoke：对 3–5 组随机 (x,φ)，`max|Δreward| < 1e-4`。kernel 级已证（diff 2e-11），这步是端到端确认，**解锁 A2/DiffuseBot 的加权 actuation 路径**。
  - 喂：贡献1 设计空间对齐的正确性。

---

## Phase G2 — Hot-loop wiring（核心；共享 `mrmfmbd_mbd_jax.py` 的 `_make_block_runner`/`step`）
> 顺序重要：四个子项都改同一条 scan。建议按 a→b→c→d，每个 wiring 后立刻 smoke 再叠下一个。所有新逻辑都已在 CPU 子包里写好+测过，这里是**接线 + JAX 镜像 + rollout 验证**。
>
> **状态（2026-06-17，pip A5000）：G2a–d 全部 wired + smoke 通过，legacy 路径 byte-identical（旧 smoke return 精确复现 0.20001652836799622，43 单测全绿）。** 改动全在 `backends/mrmfmbd_mbd_jax.py` + `experiments/framework/baselines/mrmfmbd_baseline.py`，全部 opt-in。smoke 配置在 `configs/soft_robot/co_design/smoke/g2{a,b,c,d}*.yaml`，验证脚本在 `scripts/tasks/soft_robot/co_design/smoke/`。

> ### ⚠️ 执行发现（G2c — 关系到贡献2/Table 4）
> **问题：当前 fidelity 轴选错了。** 实现里的 fidelity ladder 是**按 episode 长度**（`FIDELITY_STEPS={0:30,1:100,2:200}` env-steps）。但在 crawling reward（终值由 `100·final_disp` 主导）下，短 episode 与 200-step reward **反相关**（实测 corr(30,200)≈−0.4，corr(100,200)≈0）→ control-variate 的低保真是个**坏 proxy**，CV 在这个轴上**毫无增益甚至更差**（实测 CV MSE 0.115 > 单保真 0.078，lo/hi corr −0.43）。
>
> **建议（已部分实现）：fidelity 轴改用 paper plan 本意的"按 substeps/物理分辨率"**——同样 episode 长度，用更粗的 MPM `dt`（更少 substep）。实测 coarse `dt=1e-3`（8 substep vs fine 16，**便宜 2×**）与 fine reward **corr +0.66** → CV **跑赢单保真 3.8×**（MSE 0.072 vs 0.273，matched budget）。已加 `cv_low_dt` 旋钮（>0 走物理分辨率轴），g2c smoke 默认用它。**Table 4 的主 fidelity 轴应是物理分辨率（substep/dt），不是 episode 长度。** 更粗（`dt=1.5e-3`，5 substep）corr 掉到 0.15（逼近 CFL 上限，数值失稳），所以 coarse 档位别太激进。
>
> **✅ 已补（2026-06-18）：BudgetDual 自适应闭环现已实现并验证。** `cv_adaptive` 旋钮打开后，高保真子集 K 不再是固定 `cv_subset_k`，而是**逐 fidelity block 用 `estimator_system.budget.optimal_subset_size(B_eff, M, c_lo, c_hi_block)` 按算力预算算可负担的 K**，`BudgetDual` ν 用每 block 的 realized-cost 反馈把成本自调回 B̄（`B_eff = B̄/(1+ν)`）。host-side 闭环（cv_k 是 python int，block runner 按 (steps,cv_k) 缓存 → JIT 安全）。验证：`scripts/.../co_design/smoke/g2c_adaptive_budget.py` —— Part A 纯逻辑（便宜 block K=16/16、精细 block 砍到 8/16；紧预算下 ν=0.438 自调节）；**Part B 真 backend 端到端**（fid0/1 K=8/8、fid2 精细 K=4/8，distinct=[4,8]，legacy 路径 adaptive=False 不变）。诊断进 `cv_adaptive_summary`（per_step_K/nu/realized）+ bridge_history（`cv_k`/`cv_nu`）。剩 Table 4 的"出数"sweep（adaptive vs fixed-low/mid/high）。

- [x] **G2a A2 decoder 插 rollout 前**（Stage 2 收尾）✅ smoke 通过（x-block=8 latent，decode occ∈[0.2,1]，reward 有限；legacy 精确复现）
  - 改：`MBDConfig` 加 `morph_latent_dim` / `morph_decoder_path` 旋钮；`mrmfmbd_baseline.py` 用 `MorphDecoder.load(path, cfg)` 构造、传入 backend；`_rollout_and_marginalize` 里 x-block 维度=latent，rollout 前 `x_morph = decoder.decode(w)`（已 jittable）。
  - ◻ smoke：tiny（K5,M4,C1）+ `morph_latent_dim=8` + `data/morph_decoders/loco_cpu` → 无报错；x-block 维度=8；decode 出的 occupancy ∈ [x_lo,x_hi]；reward 有限。
  - 喂：贡献1 headline joint MF/MB、Table 1 Ours、Table 3。
- [x] **Task-2 DiffuseBot 连续 actuator(+stiffness) co-design 场对齐**（2026-06-18，gradient-free）✅ A/B/C smoke 全过
  - 多头 decoder `g(w)→{occ, actuator(n_voxels×K softmax), stiffness}`（`morph_system/decoder.py` `init_params_mh`/`decode_full`/`decode_full_batch`/`train_vae_mh`；`specs.py` 加 `decode_actuator`/`decode_stiffness`/`n_actuators`/`e_lo`/`e_hi`）。
  - 解出场 → `scene.voxel_id` gather 成 per-particle → override Stage-1 字段：`rollout_return(_batch)` 加 `actuator_weight_voxel`/`E_voxel`（None→legacy，vmap in_axes 兼容单头）。
  - backend `_decode_morph` 穿接 CV 高/低保真两路（旋钮 `morph_decoder_actuator/stiffness` 默认开、逐头 ablate）；baseline 从 `decoder_config.json` 读多头字段。
  - 数据：`build_morph_dataset`（per-voxel actuator 直方图 + 归一 stiffness）；`train_morph_ae.py --multihead --n-actuators 10` → `data/morph_decoders/loco_cpu_mh`（occ MSE 8e-4，actuator argmax acc 0.40）。
  - 验证：`scripts/.../co_design/smoke/task2_actuator_codesign.py`（A 真放置场 8-10/10 distinct；B 场改 reward Δ0.072；C e2e actuator ON≠OFF，legacy occ-only 无头）；25 单测 + g2c-adaptive CV 路径回归全过。
  - ⚠️ 当前 decoder 仅 30 资产/1500 步训练 → actuator 场未调优（Part C ON 暂比 OFF 差）；"提升 reward"靠训好 decoder + MBD 对 w 出数。喂：贡献1/3、Table 1 Ours、Fig pipeline。
- [x] **G2b prior log_prob 折进权重**（Stage 2 收尾）✅ smoke 通过（权重偏低‖w‖，ESS 健康，reward 不比关时差；regression 精确）
  - 改：w-block prior 设 N(0,I)（latent 的 `x_mean=0,x_std=1`）；`_weighted_mean` 的 `log_w` 加 `theta_prior.log_prob_batch`（已算好、当前没用）；旋钮 `use_prior_weight`。
  - ◻ smoke：开 `use_prior_weight` → 权重偏向低 ‖w‖ 候选；ESS 合理；tiny reward 不比关时差。
- [x] **G2c CV 估计器 + 解非递减 + dual-ν**（Stage 3 收尾，核心 ML）✅ CV+relax-raise wired；物理分辨率轴下 variance 3.8× 优于单保真（见上⚠️发现）；**dual-ν/BudgetDual 自适应 K 分配已实现**（`cv_adaptive`，per-block `optimal_subset_size`+ν 闭环，smoke `g2c_adaptive_budget.py` Part A+B 全过）
  - 改：把 `estimator_system/control_variate.py` 的 numpy 逻辑 JAX 镜像进 scan（low-fi 全 M + hi-fi 子集校正，替换"选一层 ℓ"）；移除/放宽 `mbd_jax.py:336` 的非递减 `raise`；接 `BudgetDual`（`nu_max` 旋钮）。
  - ◻ smoke：tiny + `method=control_variate` → 跑通；每步用到两层 fidelity；记录的 score 方差 < 单保真度（matched budget）；fidelity 可非单调。
  - 喂：**贡献2（核心）**、Table 4、Fig fidelity。
- [x] **G2d CVaR 接 marginalizer 分派**（Stage 4 收尾）✅ smoke 通过（worst-mode +0.905 vs reward；q 集中最差 regime；default 路径精确）
  - 改：`mbd_jax.py` 的 marginalizer 分派加 `regime_posterior_mode=="cvar"` → `cvar_marginalize_jax`（已写）；`MBDConfig` 加 `cvar_alpha` 旋钮（类比 `risk_temperature`）。
  - ◻ smoke：tiny + `regime_posterior_mode=cvar, cvar_alpha=0.3` → 跑通；worst-mode return 优于 `reward` 模式；q 集中低-reward regime。
  - 喂：贡献4、Fig regime、Table 1 robustness 列。

---

## Phase G3 — Baseline + 任务 smoke
- [x] **G3.1 DiffuseBot 梯度 smoke**：✅ 通过。`configs/soft_robot/co_design/smoke/g3_1_diffusebot.yaml`（n_iters=20, lr=5e-3, h=64）→ best return 0.0004→1.104，grad norm 1.03 finite（无 NaN/Inf），theta finite。短 h + 小 lr 让 contact 梯度良态。喂：贡献1（gradient-free vs gradient-based）、Table 1 "DiffuseBot-style" / "first-order ours"。
- [x] **G3.2 Carry 任务接线（wiring 完成；physics 待开发）**：✅ **wiring 全通**——`adapters.py` 加 `carry` 分派→`rollout_return_carry_batch`；`tasks/regime.py` 加 `task="carry"`（train 36 / test 60 regimes，`horizontal_only=False`）；`jax_mpm_evaluator.py` 加 `train_carry`/`test_carry`；task domain `_TASKS`+`carry`（num_modes=36）；`configs/soft_robot/co_design/main_v2/carry.yaml` + smoke `g3_2_carry.yaml`。tiny co-design 端到端跑通（finite reward），legacy 路径精确不变，43 单测全绿。
  - ⚠️ **carry physics 待开发（非 wiring 问题）**：smoke 的"好体托运 / 坏体掉落"区分**做不出来**——manipuland 的 y 在 init 时**自动 snap 到 terrain（地板），不是 body 顶**，所以物体永远落地（`G_T=0`，无区分）；把物体 init 到 body 上方/内部则**接触爆炸**（obj_y→2729，越出 [0,1] 域）。`carry_reward` 本身正确（4 单测过），缺的是**让物体稳定停在软体上的 manipuland 耦合 + 几何**（object-on-body init + 稳定竖直接触，或改成 scoop/lift 表述）。这是任务开发，不在"接线"范围内。**建议：carry 进主表前需先补这块**；push（horizontal_only=True）不受影响，可先用 push 撑 loco-manip 表。
  - 喂：loco-manip 表、贡献1/3/4 showcase。

---

## Phase G4 — Asset banks（关 Gap A）
- [x] **G4.1 clone 3D priors**：✅ TripoSG repo clone 到 `third_party/morphology_priors/triposg/repo`，权重 `VAST-AI/TripoSG`(7.5G) 下到 `repo/pretrained_weights/TripoSG`。image→mesh 推理跑通（fp16/50步~12s，watertight mesh ~285万顶点；脚本 `repo/minimal_infer.py`、`batch_infer.py`）。
- [x] **G4.2 每 prior 建 bank**：✅ `data/asset_banks/triposg`（8 assets）robotize **RobotizationSuccess 87.5%**（7/8，1 个 mesh 太小 <64 cells；远超 ≥30% target）；manifest + robotized/*.npz 生成；triposg body 下游 GPU rollout reward +0.50 finite（sim stable）。
- [x] **G4.3 （可选）重训 decoder**：✅ 合并 `data/asset_banks/mixed`（30 random_shapes + 7 triposg = 37 occupancy）→ `train_morph_ae.py` → `data/morph_decoders/mixed`（MSE 0.0011）。比 loco_cpu 更丰富（含真实 3D-gen 形态），可喂 G2a 的 `morph_decoder_path`。
  - 喂：Table 3 prior 横评、更强的 A2 先验。

> ### ⚠️ 执行发现（G4 — TripoSG 集成的坑与隔离方案，关系到可复现）
> **TripoSG 是 image-to-3D（不是文本→3D）**：主入口 `pipe(image=img_pil)`，`build_asset_bank --prior triposg` 的 text-prompt 接口对不上 → 上面用例图(`assets/example_data/*.png`)直接喂、绕过了 adapter。**所以 G4 证明的是"真实 3D-gen prior → robotize → 可仿真软体"的管线可用性，不是最终"机器人形态 bank"**——后者需一组机器人形态图像（手工收集 or text-to-image 生成），属数据工作。
> **依赖冲突必须隔离**：TripoSG pin `numpy==1.22.3`（会搞坏 JAX/MPM 的 numpy 2.2.6）→ 用 `--system-site-packages` venv `/root/triposg_venv`（继承系统 torch/numpy，只补 transformers/peft/typeguard/diso）。`diso` 需 nvcc 编译（CUDA 12.1 OK）。系统 `torchaudio 2.2.0` 与 torch 2.10 ABI 冲突、`.so` 抛 OSError 被 diffusers 当致命 → 在推理脚本顶部**注入 torchaudio stub 模块**（带 `__spec__`）+ venv 内降 `diffusers==0.32.0`/`transformers==4.49.0`。**主环境零改动**（43 单测 + MPM 不受影响）。
> adapter（`priors/triposg.py`）若要进 `build_asset_bank` 主流程，需改成 image 输入（且在隔离环境跑）；当前留作后续。其余 prior（hunyuan3d/trellis/craftsman）同法可加，进 Table 3。

---

## Phase G5 — 渲染器（关 Gap B）
- [x] **G5.1 写 `scripts/visualizations/soft_robot/co_design/render_soft_robot_checkpoints.py`**：✅ 用 matplotlib Agg 离屏 3D scatter（pyvista 没装且无显示器，改用 matplotlib 更稳）。re-rollout theta_history 选定 checkpoint，`rollout_with_positions` 取 `x_hist`（每帧粒子位置），按 `actuator_id` 着色，occupancy 过滤 + 共享坐标（体现前移）。产出 `results/soft_robot/co_design/figures/evolution_rest.png`（形态：满 box→稀疏分化）/ `evolution_final.png`（locomotion）。注：theta_history 是 z-sym optimizer 空间（24 x_half+80 phi），需 mirror 成 48。
  - **bonus 渲染脚本**：`render_soft_robot_gif.py`（单 body 爬行 gif）、`render_compare_gif.py`（多 checkpoint 并排，展示 iter 0→99 学会爬行）、`render_asset_bank.py`（robotized 形态网格 triposg vs random_shapes）、`render_meshes.py`（TripoSG 原始 mesh surface）。产出 `results/soft_robot/co_design/figures/{crawl_best,crawl_compare}.gif`、`asset_banks.png`、`g2d_{cvar,reward}_evo.png` 等。
  - 喂：Fig evolution（DiffuseBot 风格）、loco gif、Table 3 prior 质性网格。

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
