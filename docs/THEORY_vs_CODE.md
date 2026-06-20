# THEORY ↔ CODE 对照：MR-MF-MBD（Mode-Robust Multi-Fidelity Model-Based Diffusion）

**目的**：把最新理论 `new_version.txt`（"New Co-Design"）逐方程对到当前实现
`genedynamics/solvers/single/mrmfmbd/`，标注一致 / 部分一致 / 不一致，并给出修复或改写
claim 的清单。旧版报告 `_Project__Mode_Robust_Multi_Fidelity_Model_Based_Diffusion_for_Soft_Robot_Co_Design.pdf`
（下称 *旧 PDF*，IEEE 8 页，Algorithm 1 / Eq. 17–23）作参照系，用来区分"新理论新增"与"旧已有"。

**核对日期**：2026-06-20  
**生产引擎**：`backend: mbd` → `MRMFMBDBackendMBD`（`backends/mrmfmbd_mbd_jax.py`），所有 `configs/soft_robot/co_design/main_v2/*.yaml`（除 `shac_only_crawling`）都走这条。

---

## 0. 总评

| 模块 | 理论（new_version.txt） | 实现状态 |
|---|---|---|
| 联合扩散内核 + MCSA/SNIS 分数 | §III joint MBD | ✅ 结构对（实用「跟踪均值」写法，比旧 PDF 更贴新理论的联合视角） |
| 风险敏感 regime 目标 ρ_ℓ + 后验 q(m\|z) | §IV-A Eq. risk/posterior | ✅ 公式逐字一致；⚠️ 主力 crawling 配置默认没开 |
| 贡献②：预算化多保真度（自适应保真度选择 + νC 惩罚 + 对偶） | §IV-B Eq. fidelity_choice/value/weight/dual | ❌ **未落地**：代码是固定阶梯 + control-variate；对偶 inert；生产里 CV 也没开 |
| learned controller D_β(c,x) + 环内 SHAC 精修 | §IV-A learned-controller / §V | ❌ 控制器只有正弦开环；SHAC 是事后 top-K 精修，非环内、proximal 不进权重 |
| 联合 MF 先验加权 p_θ^MF(z^MF) | §III joint target | ⚠️ 仅高斯隐变量先验代理，且生产配置 `use_prior_weight=False` → 权重里没有 |
| 有效性指示 I(z) | Eq. validity_indicator | ⚠️ 软化成 TV 惩罚 + robotize 单连通闸，非 {0,1} 乘子 |
| 高保真认证 ρ_H | Eq. final_certification | ⚠️ 用跨 mode 均值 return 选优，非风险敏感 ρ_H |

**图例**：✅ 一致 ・ ⚠️ 部分一致/默认关闭 ・ ❌ 不一致/缺失

**一句话**：扩散内核与风险目标对；**贡献②（多保真度）与新旧两版论文都不符**，是头号问题；新理论新增的 learned controller / 联合 MF 加权 基本未实现或默认关。

---

## 1. 符号对照表

| 理论符号 | 含义 | 代码符号 / 位置 |
|---|---|---|
| `z = (z^MF, z^MB)` | 联合 co-design 变量 | `theta = [x\|phi]`，`ThetaParametrization`（`theta_prior.py`） |
| `z^MF = (ξ, η)` | 形态隐 + robotize 变量 | x-block = 形态隐变量 `w`（`morph_latent_dim`），decoder `g(w)`（`morph_system/decoder.py`） |
| `z^MB = φ` | 控制器变量 | phi-block（正弦/开环参数；**无** `D_β`、无 `c`） |
| `x = R(G_θ(ξ), η)` | robotization 推前 | `MorphDecoder.decode`（`_expand_x_np` / `_decode_morph`，`mrmfmbd_mbd_jax.py:419-633`） |
| `ᾱ_k, σ_k` | DDPM 调度 | `self._sigmas`（`mrmfmbd_mbd_jax.py:278-285`），单一调度（非 MF/MB 两套） |
| `Ω_k(z_k)` | clean 提议分布 | `_propose_impl` / `_propose_inline`（`:359-363`,`:607-610`） |
| `w_j, w̄_j` | 重要性权重 | `_weighted_mean` 里的 `log_w` / `softmax`（`:712-721`） |
| `z̄_k` | 加权 clean 均值 | `Ybar_inner = Σ w_m Y0s_m`（`:720`） |
| `Ŝ_k` | MCSA 分数 | 隐式（代码直接置 `Ybar ← z̄_k`，不显式构造 Ŝ_k） |
| `ρ_ℓ(z)` | 风险敏感目标 | `risk_sensitive_marginalize_jax` 返回的 `rho_m`（`regime_posterior.py:42-69`） |
| `q_ℓ(m\|z)` | regime 后验 | 同上的 `q_mc` |
| `ℓ*_k` | 自适应选中的保真度 | **无**（固定 `fidelity_ladder(kprime)`，`:393`） |
| `V̂_k(ℓ)` | 保真度价值 | **无**（`upgrade.py` 有零件但未接入） |
| `ν_k` | 算力对偶 | `BudgetDual`（`budget.py`）/ `alm_nu_schedule`——均 inert 或只调 CV 的 K |
| `C_ℓ` | 保真度代价 | `FIDELITY_STEPS[fid]`（env 步数）/ substeps，仅用于 CV 预算与 diagnostics |

---

## 2. 逐方程对照

### 2.1 问题设定与联合变量（§Preliminaries）

| 理论 | 实现 | 判定 |
|---|---|---|
| `eq:ideal_codesign_objective` 鲁棒目标 `max E_{m~p(m)}[R_H]` | 由 regime 边缘化近似（见 2.4） | ✅ |
| `eq:robotization` `x=R(G_θ(ξ),η)` | `w → MorphDecoder.decode → occupancy(+actuator/stiffness)`，`_decode_morph`（`:624-633`） | ⚠️ decoder 是自训 VAE，不是预训练 3D 生成模型 G_θ；ξ↔w 仅类比 |
| `eq:morphology_pushforward_prior` `p_θ^MF=(R∘G_θ)#p_ξ` | 用 `w~N(mean,std)` 的高斯隐先验代理；3D-prior 仅在 **init** seed（`codesign.py:133-149` `morph_init_path`） | ⚠️ |
| `eq:sinusoidal_controller` 正弦控制器 | ✅ 即生产控制器（SinWaveOpenLoop） | ✅（这是"第三特例"） |
| `eq:controller_decoder` `ϑ=D_β(c,x)` + `eq:learned_controller` 闭环 π | **未实现**（无解码器、无闭环策略、无 e_x 形态嵌入） | ❌ |
| `eq:joint_codesign_variable` `z=(z^MF,z^MB)` | `theta=[w\|phi]` 联合 | ✅ |

> 注：旧 PDF Algorithm 1 是**非对称**——外层 `for n=1..Nx` 采样形态 `x_n~p_θ`、只对 φ 扩散。
> 新理论要 x、φ **联合扩散**。当前代码确实把 `w` 和 `phi` 一起加噪/去噪 → **这一点代码比旧 PDF 更贴新理论**。

### 2.2 联合 MF/MB 扩散（§Joint Model-Free Model-Based Diffusion）

| 理论 | 实现 | 判定 |
|---|---|---|
| `eq:joint_mf_mb_target` `p*∝p_θ^MF · p_0^MB · V` | 权重 `softmax((R−mean)/(std·T))`，**默认无** p_θ^MF 项（`use_prior_weight=False`） | ⚠️ |
| `eq:joint_forward_process` 前向加噪 | DDPM `σ_k=√(1−ᾱ_k)`，`:278-285` | ✅ |
| `eq:two_schedule_forward_process` MF/MB 两套调度 | 单一调度 | ⚠️ 理论允许，未用 |
| `eq:score_posterior_mean` `S_k=−z_k/(1−ᾱ)+√ᾱ/(1−ᾱ)·E[z_0\|z_k]` | 隐式：直接用加权均值当 `E[z_0\|z_k]` | ✅（语义对） |
| `eq:clean_joint_proposal` `Ω_k=N(z_k/√ᾱ_k,(1/ᾱ_k−1)I)` | `Y0s=Ybar+σ_k·scale·ε`，`σ_k=√(1−ᾱ_k)` 截断 `sigma_max=0.15`，中心=running mean | ⚠️ 实用「跟踪均值」写法：中心与方差都不是字面 Tweedie |
| `eq:joint_importance_weight` `w_j∝p_θ^MF·p_0^MB·V` | `_weighted_mean`：`log_w=(R−mean)/(std·T)(+prior)`，`:712-721` | ⚠️ 标准化 softmax 温度，非字面 `exp(ρ/τ)` |
| `eq:joint_clean_mean` `z̄_k=Σ w̄_j z̃_j` | `Ybar_inner=einsum(w, Y0s)`，`:720` | ✅ |
| `eq:joint_mcsa_score` / `eq:joint_reverse_update` `z_{k-1}=(1/√α_k)(z_k+(1−ᾱ_k)Ŝ_k)+σ_kε` | `Ybar←Ybar_inner + τ_k·scale·ξ`，`τ_k=σ_k·0.1`，`:744-746` | ⚠️ 直接置 clean mean + 小探索噪声；**无** √α 重标定、无 σ_k ε 完整步 |

**小结**：扩散内核 = 原版 M3BD 的实用形式（新理论 §结尾"第三特例"明确说会 recover 它）。若论文正文照抄 `eq:joint_reverse_update`，需注明代码用的是 mean-tracking 变体。

### 2.3 Co-Design as Inference（§Co-Design as Inference）

| 理论 | 实现 | 判定 |
|---|---|---|
| `eq:physical_potential_general` `Ψ=I(z)·exp(ρ/τ)` | reward 进 softmax 权重；无显式 exp(ρ/τ) 乘子 | ⚠️ |
| `eq:validity_indicator` `I(z)=𝟙[connected/actuated/simulatable]` | 软化：`shape_weight·TV(occupancy)` 罚（`_shape_penalty`,`:665-708`）+ robotize `require_single_component` 闸 | ⚠️ 软惩罚，非硬指示 |
| `eq:conditional_controller_target` 条件控制器特例 | 可由"固定 w + 只扩散 phi"实现，但生产是联合扩散 | ✅（特例可达） |

### 2.4 风险敏感 regime 分数（§Methodology-A）★核心一致点

| 理论 | 实现 | 判定 |
|---|---|---|
| `eq:regime_risk_objective` `ρ_ℓ=−τ_r·logΣ_m p(m)exp(−R/τ_r)` （= 旧 PDF Eq.17） | `regime_posterior.py:66-67`：`rho_m=−τ_r·logsumexp_c(log p − R/τ_r)` | ✅ **逐字一致** |
| `eq:failure_regime_posterior` `q_ℓ(m\|z)=softmax(log p − R/τ_r)` | `:68` `q_mc=softmax(log p − R/τ_r)` | ✅ |
| τ_r→0 → min_m R；τ_r→∞ → 均值 | 极限正确（`tau_eff=max(τ_r,1e-8)`） | ✅ |
| `eq:risk_gradient` `∇ρ=Σ q·∇R` | 不需（gradient-free） | n/a |
| `eq:regime_interaction_potential/weight` | ρ_m 进同一 softmax 权重 | ✅ |

**默认值警告**：`MBDConfig.regime_posterior_mode` 默认 `"reward"`（旧的"易 regime 加权"`_s1_marginalize_jax`，`:187-202`，**非**鲁棒目标）。生产配置里：

| 配置 | regime_posterior_mode |
|---|---|
| `main_v2/locomotion_terrain.yaml` | `risk_sensitive` ✅ |
| `main_v2/push.yaml` | `risk_sensitive` ✅ |
| `main_v2/carry.yaml` | `risk_sensitive` ✅ |
| `main_v2/crawling_alm.yaml`（主力 crawling） | **缺省 → `reward`** ⚠️ |
| `main_v2/crawling_alm_shac.yaml` | `reward` ⚠️ |
| `main_v2/crawling_from_mesh.yaml` | `reward` ⚠️ |

> `cvar` 模式（`cvar_marginalize_jax`,`regime_posterior.py:72-119`）是新理论里**没有**的额外鲁棒度量——当 ablation 用 OK，别在正文当成"那个目标"。

### 2.5 learned-controller 局部精修（§Methodology-A learned-controller paragraph）

| 理论 | 实现 | 判定 |
|---|---|---|
| `eq:shac_q` 短视界 critic `Q̂_{ℓ,h}` | `_shac_refine_topk` 的 `loss_fn` 用 `rollout_h_from_state` 折扣 return（`:1063-1074`） | ⚠️ 有 h-步 return，无 critic `V_ψ` bootstrap |
| `eq:controller_latent_refinement` `c⁺=c+η∇ρ̂` | SGD on phi：`phi_j -= lr·g`（`:1096-1101`） | ⚠️ 精修 φ 本身（非 latent c via D_β） |
| `eq:shac_regime_weight` proximal 项进**每步重要性权重** | proximal 在**局部 BPTT loss** 里（`:1073`），**不进任何扩散权重** | ❌ 位置错 |
| 环内、每候选、权重前 | **事后**：reverse loop 全跑完后对最终 top-K 做一次（`:968-979`） | ❌ 结构不同 |

### 2.6 预算化多保真度（§Methodology-B）★头号不一致

| 理论（= 旧 PDF Eq.20–23 + Algorithm 1 行 13/14/17/20） | 实现 | 判定 |
|---|---|---|
| `eq:fidelity_choice` `ℓ*_k=argmax_ℓ[V̂_k(ℓ)−ν_k C_ℓ]`（明确"**不用**固定 coarse-to-fine"） | `fid=fidelity_ladder(kprime)`，**固定**几何/线性/cosine 阶梯，`_build_fid_blocks` 还**强制非递减=coarse→fine**（`:382-415`,`:393`）；posterior backend 同（`mrmfmbd_posterior_jax.py:328`） | ❌ 正好是论文说不用的固定调度 |
| `eq:fidelity_value_estimator` `V̂_k=a1·Var_j[ρ]+a2·H(q̄)+a3·Δ̂_{ℓ,H}−a4·ESS⁻¹` | **无选择器**。零件存在但未接入：`upgrade.py` 的 `ScoreGapUpgradeRule`(=Δ̂)、`ESSUpgradeRule`(=ESS⁻¹) 只在 `__init__.py` re-export，**两个 backend 都没调用** | ❌ |
| `eq:multifidelity_weight` 权重含 `−ν_k C_{ℓ*}` | `_weighted_mean` 的 `log_w` **没有** νC 项（`:715-718`）。`alm` 的 ν 调度/代价只写进 diagnostics（注释自承 "Phase 1 only wires the scaffold"，`:298-331`），且所有主配置 `nu_max=0` → inert | ❌ |
| `eq:dual_update` `ν_{k-1}=[ν_k+η(C_{ℓ*}−C̄)]_+` | `BudgetDual.update`（`budget.py:62-64`）公式对，但只在 `cv_adaptive` 下调 **CV 子集大小 K**，**不**调保真度等级，也不进权重 | ❌ 用错对象 |
| 高保真两角色（环内软校正 + 终认证） | 终认证有（`_fine_revalidate`）；环内软校正无 | ⚠️ |

**代码实际有的"多保真度" ≠ 论文**：是一套 **control-variate**（`estimator_system/control_variate.py`；`_rollout_and_marginalize` 的 CV 分支 `:685-708`）——全 M 跑低保真 + top-K 跑高保真，`R̃_m=R_lo_m+Δ̄`，`Δ̄=mean_{j∈S}(R_hi−R_lo)`。这套 CV 在**新旧理论里都没有**。而且：

```
$ grep -rnE "cv_enabled|cv_adaptive|use_prior_weight" configs/soft_robot/co_design/main_v2/*.yaml
（空 —— 生产配置一个都没开）
```

→ 生产跑的多保真度 = **固定阶梯 + 惰性对偶**，连 CV 替身都没启用。这与旧 PDF 的 Q3/Table II 主张（"ν_k 轨迹"、"固定 coarse-to-fine 无法表达的自适应行为"）**直接冲突**。

### 2.7 Putting Everything Together（§Putting Everything Together）

| 理论 | 实现 | 判定 |
|---|---|---|
| `eq:final_proposal/decode` 提议+解码 | ✅ `_propose_inline` + `_decode_morph` | ✅ |
| `eq:final_fidelity_choice` 选 ℓ* | ❌ 固定阶梯（见 2.6） | ❌ |
| `eq:final_risk_eval` ρ_{ℓ*} | ✅ 在选中（固定）保真度上算 ρ | ✅（在错保真度上） |
| `eq:final_learned_controller_update` 环内 c⁺ | ❌ 见 2.5 | ❌ |
| `eq:final_importance_weight` `w∝p_θ^MF·p_0^MB·I·exp(ρ/τ−λ‖Δ‖²−ν C)` | `log_w=(R−mean)/(std·T)`：**缺** p_θ^MF（默认）、**缺** λ‖Δ‖²、**缺** νC | ❌ 三项都缺 |
| `eq:final_clean_mean/score/reverse` | ✅ mean-tracking 变体（见 2.2） | ⚠️ |
| `eq:final_dual_update` | inert | ❌ |
| `eq:final_certification` `argmax ρ_H` + `eq:high_fidelity_risk` 风险敏感 ρ_H | `_fine_revalidate` 按**跨 mode 均值** `np.mean(res.returns)` 选优（`:1156`） | ⚠️ 认证用均值，非 ρ_H |

---

## 3. 三个特例的对应（new_version.txt 结尾）

| 新理论特例 | 是否=当前实现 |
|---|---|
| ① `z^MF` 采样后固定 → 条件控制器 MBD | 可达（固定 w，只扩散 phi） |
| ② 风险目标→均值 reward 且固定保真度 → nominal fixed-fidelity MBD | **≈ 当前生产默认**（`regime=reward` + 固定阶梯），即论文的 baseline，不是 full method |
| ③ φ→正弦 + 关 actor-critic 精修 → 原 sinus M3BD | **= 当前实现的内核** |

> 也就是说：**当前 shipped 代码落在特例②/③之间**，离 new_version.txt 的 full method（联合 MF 加权 + 自适应保真度 + learned controller 环内精修）还差贡献②与 learned-controller 两大块。

---

## 4. 修复 / 改写清单（按优先级）

1. **【必做·贡献②】落地自适应保真度选择**，替换固定阶梯：
   - scan 内对每步算 `V̂_k(ℓ)`（先接现成 `upgrade.py` 的 Δ̂ 与 ESS⁻¹，再补 `Var_j[ρ_ℓ]`、`H(q̄_ℓ)`），按 `argmax_ℓ[V̂_k(ℓ)−ν_k C_ℓ]` 选 ℓ；
   - 把 `−ν_k C_{ℓ*}` 真正加进 `_weighted_mean` 的 `log_w`；
   - `BudgetDual` 用 `eq:dual_update` 调 ν（对象改成保真度代价，而非 CV 的 K）；
   - 旧固定阶梯/CV 保留为 ablation。
   - **替代方案**（若不改代码）：把论文 contribution② 改写成"固定阶梯 + control-variate 多保真度"，删掉 Eq.20–23/ν_k 叙述与旧 PDF Q3 的自适应主张。
2. **【主表口径】** headline 鲁棒结果跑在 `risk_sensitive` 上（`crawling_alm.yaml` 现为默认 `reward`）。
3. **【认证】** `_fine_revalidate` 选优改成风险敏感聚合 ρ_H（`eq:high_fidelity_risk`），而非跨 mode 均值。
4. **【联合 MF 加权】** 主配置打开 `use_prior_weight`，并把 p_θ^MF 接成真实 3D 生成先验的 score/log-prob（现在只是 `w` 的高斯）。
5. **【learned controller】** 若要支撑新理论主线：加 `D_β(c,x)` 解码器 + 闭环 π + 环内 SHAC（proximal 进每步权重）。否则论文须明确限定在"正弦控制器 + 形态隐高斯先验"特例。
6. **【字面公式】** 若正文用 `eq:joint_reverse_update`/`eq:clean_joint_proposal`，注明实现是 mean-tracking 变体（中心=running mean、`σ_k=√(1−ᾱ_k)` 截断、`+τ_k` 小噪声）。

---

## 5. 关键代码位置索引

| 主题 | 文件:行 |
|---|---|
| MBDConfig 全旋钮 | `backends/mrmfmbd_mbd_jax.py:63-170` |
| DDPM 调度 / τ_frac | `:278-296` |
| alm ν 脚手架（inert） | `:298-331` |
| 固定保真度分块 | `:382-415` |
| reward 边缘化（legacy） | `:187-202` |
| 风险/CVaR 边缘化选择 | `:635-640` → `mode_system/regime_posterior.py:42-119` |
| CV 多保真度分支 | `:674-708` |
| 重要性权重 log_w | `:712-721` |
| 反向步（mean-tracking） | `:723-775`（噪声+clip `:744-746`） |
| 固定阶梯主循环 + cv_adaptive | `:818-1019`（block 循环 `:871-916`） |
| SHAC 事后精修 | `:1021-1123` |
| 高保真认证（均值） | `:1125-1160`（均值 `:1156`） |
| 配置→MBDConfig 映射 | `codesign.py:187-259` |
| 3D-prior init seed | `codesign.py:133-149` |
| 固定阶梯工厂 | `fidelity_system/ladder.py` |
| 未接入的升级规则 | `fidelity_system/upgrade.py` |
| 预算/对偶 | `estimator_system/budget.py` |
| CV 估计器 | `estimator_system/control_variate.py` |
| posterior backend（也用固定阶梯） | `backends/mrmfmbd_posterior_jax.py:328` |
