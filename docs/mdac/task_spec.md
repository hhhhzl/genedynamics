# MDAC 任务 Spec — Exp I 机械臂 S1–S4 / Exp II 人形 H1–H4

> 重建 env 前的任务设计,**已按用户拍板锁死(2026-06-23)**。核心原则:第一版**不上 26D full
> primitive**。相对 DIAL-MPC torque-level 的优势是"更低维、更结构化、接触语义化的 manifold
> primitive",不是更复杂的 action space。所以主实验 = **10D arm + 12/15D humanoid box-unjamming**;
> 26D full / 完整 WBC 写成 §3 general formulation / future,不作第一版默认 active primitive。
> 约束/指标编号对到 idea.txt。

---

## 0. 公共约定(两实验共享)
- 采样对象 = lower-control 序列 `U = u_{0:H-1}`;状态由闭环 rollout `s_{h+1}=F(s_h,u_h;c)` 得到,**不单独采状态**。
- 位置-刚度原语 `K_h = exp(S_h)`,在对称阵 `S_h` 上采样;`svec(d×d)` 维度 `= d(d+1)/2`。
- 约束 `C=[h; g+½μ²]`;soft-feasibility 残差 `r=[h; [g]_+]` 喂 AL(已接);切投影 `P_M^G`+回缩(已接);耦合退火非单调 overlay(已接)。
- 每档统一 `(H,K,M)` 预算,baseline/ablation 同预算(`assert_fair`)。

---

## 1. Exp I — 表面接触扫描(7-DoF Franka Panda)【主实验】

### 1.1 primitive `u_arm = (Δξ, Δη, Δψ, S_t, F_n^d) ∈ R^10`  ✅锁死

| 块 | 含义 | 维度 |
|---|---|---|
| `(Δξ, Δη)` | 表面坐标增量 | 2 |
| `Δψ` | 绕法向 yaw 增量 | 1 |
| `S_t` | **3×3 translational SPD** 刚度 log-svec,`K=exp(S)∈S³_++` | **6** |
| `F_n^d` | 期望法向力 | 1 |
| **action_size** | | **10** |

- orientation 刚度 `K_R` **固定**(不进 action);probe 姿态由法向定 `R_d e_z = -n_s(ξ,η)`。
- 6×6 full task-space 刚度(svec 21 → action 25)**只放 appendix 做 scalability ablation**,非主实验。

### 1.2 底层 `π_low`(Cartesian impedance,简化版先行)✅锁死
`p_t^d = p_s(ξ,η)+d·n_s`;`R_t^d e_z = -n_s`;`F_t = K_t(p_t^d-p_t)+D_t(ṗ_t^d-ṗ_t)+F_n^d n_t`,`K_t=exp(S_t)∈S³_++`;`τ_t = J(q)^T F_t + τ_null`。orientation controller 固定 gain。

### 1.3 曲面族 S1–S4 ✅锁死(NURBS 底座给 S2/S3/S4;S1 解析)

底座:**5×5 NURBS 控制网**(height-field,clamped 均匀节点),`p_s(ξ,η)` + 解析法向 `n_s=(∂_ξp×∂_η p)/‖·‖`(基函数导数,不用数值差分)。patch 尺寸**≥0.4 m×0.4 m**;若用小 patch(0.2×0.3)bump 幅值降到 0.02–0.03 m。

| 族 | 几何 | 高度幅值 | 用途 |
|---|---|---|---|
| **S1** | 平面 / 圆柱(解析) | `A=0` | sanity check |
| **S2** | smooth convex NURBS(abdomen/椭球样) | `A_convex ∈ [0.08, 0.12] m` | 凸面基准 |
| **S3** | bumpy NURBS = convex base + 局部高斯 bump | `A_i ∈ [-0.04, 0.04] m`(稳一点 [-0.03,0.03]) | 非凸 |
| **S4** | unseen 随机 NURBS + DR | `A_i ∈ [-0.06, 0.06] m` + 随机表面刚度/摩擦 | 泛化/未见 |

S3/S4 bump 生成:`z(x,y)=z_base(x,y)+Σ_i A_i exp(-‖(x,y)-c_i‖² / 2σ_i²)`(写进 5×5 控制点高度后 NURBS 插值,保证法向光滑解析)。

### 1.4 约束向量 ✅锁死

| 符号 | 定义 | 类型 | 维度 | 现状 |
|---|---|---|---|---|
| `h_surf` | `p_ee - p_s(ξ,η) - d·n_s` | eq | 3 | ✅ |
| `h_normal` | `R_ee e_z + n_s(ξ,η)` | eq | 3 | ✅ |
| `g_force` | `[F_n - F_max ; F_min - F_n]` | ineq | 2 | **新增(现只在 unpack clip)** |

`constraint_residual → (h=[h_surf;h_normal](6), g=g_force(2))`;clean-state 解析 `J_C`(不穿 mjx)。

### 1.5 域随机化(仅 S4)✅锁死
表面刚度 `k_surf ∈ [2e3, 1e4] N/m`;接触摩擦 `μ ∈ [0.3, 1.0]`;控制网 seed = held-out。

### 1.6 指标(idea Exp I)
surface tracking error、normal alignment error、force tracking error、force violation rate、force CVaR95、contact loss rate、path completion、coverage ratio、stiffness smoothness、energy、runtime。
> 通用库已有大部分;**新增** `surface_tracking_error`(‖h_surf‖)、`normal_alignment_error`(‖h_normal‖)两个通用指标。

### 1.7 train/test
S1/S2 固定典型面;S3 训练 seed 集随机 NURBS;**S4 held-out seed(从不参与调参)+ DR** → 测未见/泛化(RQ4)。

---

## 2. Exp II — 人形接触流形(Unitree H1)【主实验 = H4 box-unjamming】

> ⚠️ primitive 从"19 维关节 PD"**改成接触语义化的分层 primitive**;底层用**简化 `π_low`**(固定站立 + 手 Cartesian 阻抗 + 简单平衡器),**不上完整 WBC**(完整 WBC = §3 future)。
> 接触面 `j ∈ {rear, left, right}`(one-hot=**3**,不含 front;front=拉箱/绕前,复杂化,留 future)。

### 2.1 分层 primitive ✅锁死(主 benchmark 默认 active 见加粗)

| Task | primitive | dim |
|---|---|---|
| Arm surface | `(Δξ,Δη,Δψ,S,F_n)` | 10 |
| **Humanoid H1/H2** | `(j,a,b,S_hand,F_n)` | **12** |
| **Humanoid H4-A**(固定 base/feet) | `(j,a,b,S_hand,F_n)` | **12** |
| **Humanoid H4-B**(允许小 base 调整,主文优先) | `(v_base,j,a,b,S_hand,F_n)` | **15** |
| Humanoid H3(optional/二阶段) | `(v_base,p_foot,j,a,b,S_hand,F_n)` | 17–19 |
| general/future(§3) | `(v_base,p_foot,j,a,b,S_hand,S_body,F_n)` | 25–26 |

拆分:`j`=3(one-hot rear/left/right),`(a,b)`=2,`S_hand`=svec(3×3)=6,`F_n`=1 → 12;`v_base=(v_x,v_y,ω_z)`=3 → H4-B 15;`p_foot=(Δx_L,Δy_L,Δx_R,Δy_R)`=4 → H3 19,或 event-based swing `(Δx_swing,Δy_swing)`=2 → H3 17。
**H4-B vs H4-A**:若简化站立控制器能吃小 `v_base` 命令 → 主文用 H4-B(15);否则 H4-A(12)。

### 2.2 底层 `π_low`(简化版先行)✅锁死
- **H1/H2/H4**:`fixed double-support stance + hand Cartesian impedance + simple balance stabilizer`。下肢固定双支撑;pelvis/base 仅小范围调整(H4-B 的 `v_base`);手 Cartesian 阻抗 `K_hand=exp(S_hand)`;`F_n^d` = 推力;平衡用固定 PD / centroidal 启发式。**H4 第一版不需要 footstep WBC。**
- **H3**(二阶段):`pretrained locomotion tracker + hand impedance`(或 simple footstep generator + whole-body IK + PD)。

### 2.3 约束向量(eq:humanoid_manifold)✅锁死(标注每 level 激活项)

| 符号 | 定义 | 类型 | 维度 | 现状 |
|---|---|---|---|---|
| `h_box` | `[p_bz-z0; φ_b; ϑ_b]` 箱贴地无滚转/俯仰 | eq | 3 | ❌ 缺 |
| `h_hand` | `p_hand(q) - T_b p_face^j(a,b)` | eq | 3 | ⚠️ 粗版 |
| `h_hand_R` | `R_hand e_z + R_b n_j` 手朝向对齐面法向 | eq | 3 | ❌ 缺 |
| `h_foot^i` | `[p_fi_z; R_fi e_z - e_z; v_fi]` 支撑脚 | eq | 7×stance | ❌ 缺 |
| `g_bal` | CoM 在支撑多边形内 | ineq | 1 | ✅ |
| `g_fric` | `‖f_t‖ - μ f_n` 摩擦锥 | ineq | 1 | ❌ 缺 |
| `g_tip` | 倾覆裕度 | ineq | 1 | ❌ 缺 |

**每 level 激活**:
- **H1/H2**:`j` 固定、`p_foot` 关、双脚固定 stance。约束全集(`h_foot` 固定双支撑)。
- **H4**:`j` relaxed one-hot{rear,left,right} 激活;固定 stance(H4-B 加小 `v_base`)。核心 = 接触面选择 + hand-box 流形 + 刚度 + 力安全 + 平衡。
- **H3**:`p_foot` 激活 + stance relaxed one-hot{L,R,double},foot 切换(eq:hybrid_union)。

### 2.4 任务 level ✅锁死

| level | idea.txt | primitive | DR | 阶段 |
|---|---|---|---|---|
| **H1** | 双支撑、固定脚 | 12 | 关 | 第一版 |
| **H2** | 重箱 + 随机质量/摩擦 | 12 | **箱质量+摩擦** | 第一版 |
| **H4** | 卡箱解困 + 接触面选择 | 12(A)/15(B) | 关 | **第一版主实验** |
| **H3** | 行走推箱 + 换支撑脚 | 17–19 | 关 | optional 二阶段 |

### 2.5 指标(idea Exp II)
box-goal success rate、fall rate、box pose error、hand contact loss、hand tangential slip、friction-cone violation、tipping margin、foot slip、balance margin、force CVaR95、stiffness smoothness、energy、runtime。通用库基本齐,接 extractor 即可。

### 2.6 域随机化(仅 H2)✅锁死
箱质量 `∈ [10, 50] kg`;箱-地/手-箱摩擦 `μ ∈ [0.3, 1.0]`。

---

## 3. General formulation / future(写进 method generality,**非第一版 active**)

- **Full primitive(25–26D)**:`u_hum = (v_base, p_foot, j, a, b, S_hand, S_body, F_n)`,`j`∈{front,rear,left,right}(one-hot=4)→26;含 `S_body`(svec 3×3=6)。
- **完整 WBC `π_low`**(终局版,不作第一版依赖):
  `min_{q̈,τ,f} ‖J_hand q̈ - ẍ_hand^d‖² + ‖J_foot q̈ - ẍ_foot^d‖² + ‖h_centroidal‖² + ‖τ‖²`
  s.t. `M(q)q̈ + h(q,q̇) = Sᵀτ + J_cᵀ f`,`f ∈ K_fric`,`τ_min ≤ τ ≤ τ_max`。
- 理由:full WBC 调不稳时,实验失败难判定是 manifold diffusion 的问题还是 WBC 工程问题 → 第一版隔离掉这个风险。

---

## 4. Baseline / Ablation(idea Tab.)
- **Baseline(可公平跑的采样类)**:MPPI、DIAL-MPC、MDAC(同 `U`/同 `F`/同 `(H,K,M)`;去切投影+回缩=DIAL/MPPI)。RL 类(PPO/SAC、SRL-VIC、ATACOM、RL+CBF)需训练,排后。
- **Ablation(8 项,已在 method_registry)**:w/o stiffness、fixed stiffness、Euclidean stiffness、w/o RL prior、w/o MB rollout、w/o tangent、w/o retraction、w/o adaptive schedule。
- **RQ**:RQ1 primitive vs traj/torque;RQ2 stiffness;RQ3 tangent+retraction vs sequential filter;RQ4 MF+MB(需 RL prior);RQ5 同框架跨 arm/humanoid。

---

## 5. 构建顺序 —— ✅ 两条线已重建 + docker 验证

**Arm 线 ✅**(`core/coverage/surface_geometry.py` + `envs/domains/manipulation/panda_brax.py`)
1. ✅ NURBS 5×5 底座(`p_s`+解析法向,Cox-de Boor;单测 8/8:partition-of-unity、法向 vs FD<2e-3、jit/grad)。
2. ✅ S1 解析 / S2 凸 NURBS / S3 bumpy / S4 unseen+DR;`g_force(2)` 暴露(非空:nu=+1→力 28>f_max 20→g>0);primitive 10D(3×3 刚度 svec6)。
3. ✅ arm extractor 补 `surface_tracking_error`/`normal_alignment_error`(bind equality_residual_rms 到 h_surf/h_normal)+ force_violation_rate;**docker: ARM OK**(5 族全过、S4 DR 跨 seed 变、消融 active)。

**Humanoid 线 ✅**(`envs/domains/humanoid/box_push_brax.py`)
4. ✅ 分层 primitive `(j,a,b,S_hand,F_n)`=12 /+`v_base`=15;简化 `π_low`(右臂 Cartesian 阻抗 J^T + 其余关节 stance PD + base lean);电机=力矩,直接 pipeline_step(τ)。
5. ✅ 全约束 `h_box(3)/h_hand(3)/h_hand_R(3)/h_foot(2)`=11 + `g_bal/g_fric/g_tip`=3;clean-state geometry proxy = 常刚度+力目标流形。
6. ✅ H4 box-unjamming(relaxed face one-hot{rear,left,right})+ H1/H2(固定 rear)+ H2 DR(手摩擦+箱 frictionloss);H3 行走 = optional 二阶段(未做)。
7. ✅ humanoid extractor 全指标(goal/fall/balance/tangential_slip/contact_loss/friction_cone/tip/force_cvar/...);**docker: HUMANOID OK**(4 档全过、face select Δ=0.6、H2 DR 变、消融 active、metric plugin 端到端 14 指标 finite)。

**收尾(下一步)**:两实验 × baseline/ablation × 各档 × 多 seed → 通用指标表(RQ1–5)。run_experiment.py 需更新成跑 S1–S4 / H1/H2/H4 矩阵。

---

## 6. 锁死决策摘要
1. **Arm = 10D**(3×3 translational 刚度 svec 6);6×6 → appendix scalability ablation。
2. **Humanoid 分层**:H1/H2/H4-A=12,H4-B=15(主文优先),H3=17–19(optional),26D=general/future。
3. **接触面 `j ∈ {rear,left,right}`=3**(不含 front)。
4. **`π_low` 简化版先行**(arm Cartesian impedance / humanoid 固定站立+手阻抗+平衡器);完整 WBC=future。
5. **NURBS 5×5**;S2 凸 A=0.08–0.12,S3 bump A=±0.04,S4 unseen A=±0.06+DR;patch≥0.4 m。
6. **主实验 = 10D arm + 12/15D humanoid box-unjamming(H4)**。
