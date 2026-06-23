# MDAC 实验运行规划(arm 表面扫描 + humanoid 推箱)

> 目的:在真正开跑前,把"怎么跑、理想预期、失败怎么 debug"想清楚。**本文不跑实验**。

## 0. 诚实的现状:现在到底能展示什么

两个 env 都跑通了(docker 真 brax),指标 extractor 也通了。**Phase B 接线已全部完成**——MDAC 的 6/7 个组件在这两个 env 上都真正活跃且可消融(`test_mdac_ablation_docker.py` 验证:每个消融 vs 全量 mdac 的 plan `max|Δ| ≫ 1e-4`):

| MDAC 组件 | 受哪个 flag 控制 | 状态 | 接法 |
|---|---|---|---|
| 位置-刚度原语(K=exp S) | `use_stiffness` / `log_spd_stiffness` | ✅ active | env `stiffness_mode`(none/log_spd/euclid/fixed) |
| 软可行性 AL(soft-feasibility) | `use_soft_feasibility` | ✅ active | `build_brax_rollout_augmented` + `aug_rho` |
| 几何切投影 | `use_tangent_projection` | ✅ active | env `mdac_geometry_fn`(clean-state,无 mjx) |
| CFS 回缩 | `use_retraction` | ✅ active | genemetry `CfsRetraction` + `env.mdac_constraint` filter(**像 2GO 一样接**) |
| 耦合退火(rho↑/kappa↑) | `use_adaptive_schedule` | ✅ active | genemetry `ScheduleOverlay`(sigma↓ 归 base schedule) |
| RL prior(warm-start mix) | `use_rl_prior` | ✅ seam active | `prior=` 传入即生效(**RQ4 真数字还需训练真 PPO prior**) |
| MB rollout | `use_mb_rollout` | inert-by-design | 并行核本身就是 MB rollout 加权均值,关掉无意义 |

**结论:完整 RQ1-5 消融现在都能跑了。**字节一致门保住(seams off → mdac == DIAL,`max|Δactions|=0.00e+00`),AL 仍把违反从 2.56 降到 0.0。

`METHOD_TABLE` 里的消融(no_softfeas / no_stiffness / no_tangent / no_retraction / no_anneal / no_rl_prior)**全部真改行为**(docker 验证 active)。唯一待办:RQ4 要真数字得先训 RL prior(seam 已验证,见 §5)。

---

## 1. 跑法:轻量实验脚本(不用全套 plugin framework)

全套 `ExperimentRunner` 需要 MethodPlugin(MDAC 现在是 solver,不是 MethodPlugin)+ EnvironmentPlugin(arm/humanoid 是 env factory,不是 EnvironmentPlugin)+ obstacle/scheduler 等,接全套是额外工作量。**第一轮用一个轻量脚本**(直接用通用指标库 + aggregate):

```
for task in [arm, humanoid]:
  for method in [mdac, dial, mdac_no_softfeas]:        # 同 (M,H,K) → assert_fair
    for level in task.levels:
      for seed in seeds:
        env   = make_env(task, level)
        x0    = env.reset(seed)
        sol   = MDACSolver(env, None, jax, method=method, aug_rho=..., **CFG)
        traj  = sol.solve(x0, n_steps)                  # bridge 闭环
        rec   = task.metrics_plugin.compute(traj, env, None, None, x0=x0, planning_time=t)
        rec.update(method=method, level=level, seed=seed)
        save(rec)                                        # 立即落盘(防 OOM 丢结果)
  table = aggregate_by(records, "method")               # mean±std + per-metric CVaR95
```
docker 跑(`genedynamics/dev-cpu:torch`)。**公平性**:所有 method 用同一份 CFG(`Nsample/Hsample/Ndiffuse`)→ `method_registry.assert_fair`。

---

## 2. 运行矩阵(全部组件已接,可一次跑完整 RQ1-5)

- 任务:arm(levels: plane / cylinder / ellipsoid)、humanoid(push_dist 例如 0.4;可加 0.6)。
- 基线 + 全量:`mdac`(全开)、`dial`(=退化,字节一致自检)、`mppi`。
- 单组件消融(每个都 docker 验证 active):`mdac_no_softfeas`(RQ3-AL)、`mdac_no_stiffness` / `mdac_fixed_stiffness` / `mdac_euclid_stiffness`(RQ2)、`mdac_no_tangent` / `mdac_no_retraction`(RQ3-几何)、`mdac_no_anneal`(耦合退火)、`mdac_no_rl_prior`(RQ4)。
- 规模(先小后大):`Nsample=64, Hsample=8, Hnode=4, Ndiffuse_init=3, n_steps=6~12`,seeds=4~8;所有方法同 CFG → `assert_fair`。
- aug 扫描:`aug_rho ∈ {0(=dial), 50, 200}`。
- **唯一前置**:`no_rl_prior` 要对照出 RQ4 真数字,得先训真 PPO prior(§5);其余消融现在就能跑。

---

## 3. 理想预期(每指标方向 + 现实幅度)

**主结果表**(每任务一张,行=方法,列=指标,mean±std + CVaR95):

| 指标 | mdac(AL) 相对 dial 的预期 | 备注 |
|---|---|---|
| `violation_cvar` / `violation_rate` / `max_violation`(humanoid g_bal) | **更低** | RQ3 的软可行性部分;headline |
| `equality_residual_rms`(arm 表面跟踪 h;humanoid 手接触 h) | **更低** | AL 把解推向流形 |
| `balance_margin`(humanoid) | **更高(更安全)** | |
| `goal_error` / 任务进展 | **相当或略好** | AL 不应明显伤任务 |
| `control_smoothness` / `energy` | 相当 | |

**现实警告(重要):**
- 首轮幅度会**小**。docker AL 测试里 humanoid ‖h‖ 0.605→0.576(~5%)、arm 1.091→1.022(~6%),那是 `Nsample=64 / Ndiffuse=3 / 短 horizon / 未调参`。**首轮看"方向对不对 + pipeline 通不通",不是看大幅领先。** 幅度靠调 `aug_rho / Nsample / Ndiffuse / horizon / reward 权重` 放大。
- 退化检查:`dial`(seams off)必须 == 全开退化基线(字节一致,已验证 `max|Δactions|=0.00e+00`)。
- 每个消融 vs 全量 mdac 的**单步 plan 已验证有差**(`test_mdac_ablation_docker`),所以闭环指标出现分离是预期的;若某消融的闭环指标**完全等于** mdac,要回去查它在闭环里是否被其它项淹没。

**成功长什么样:** mdac 的 violation 列在多 seed/level 上**系统性低于** dial,且 task 成功不降;`aug_rho` 越大违反越低(到某个点饱和/伤任务)。

---

## 4. 失败模式 + debug playbook

| # | 症状 | 根因 | 怎么查/修 |
|---|---|---|---|
| 1 | 指标 NaN/inf | env 炸(mjx 不稳、impedance τ 太大、home 配置差);humanoid 有 "overflow in cast" 警告 | 逐步查 trajectory.states 是否 finite;`env.step(x0, zeros)` → 小随机 → 看 qpos;调小 `action_limit`/`aug_rho`/`d_damp`;查 overflow 警告是否污染指标 |
| 2 | mdac == dial(无分离) | AL 没活 | 确认 `constraint_residual` 返回非空 h/g;`aug_rho>0`;`method="mdac"`(不是 "dial");确认走的是 `build_brax_rollout_augmented`;打印每步 mean[g]_+ |
| 3 | AL 活但无改善 | `aug_rho` 太小 / horizon 太短 / `Nsample` 太小,找不到可行 | 扫 `aug_rho`;加 `Nsample`/`Ndiffuse`;确认违反对 aug_rho 有响应(docker AL 测试证明有) |
| 4 | 对比不公平 | 方法 (M,H,K) 不同 | `assert_fair((Nsample,Hsample,Ndiffuse), ...)`;所有方法同一 CFG |
| 5 | 方差大 / 结果跳 | 混沌(M5 教训:闭环轨迹放大 float-eps)+ seed 少 | 多 seed(≥8),报 mean±std + CVaR;别看单 seed |
| 6 | docker OOM / 被 kill | 多个 fresh-jit solver 同进程 | **顺序跑**,每个 record 立即落盘 JSON;同 method 跨 seed 复用编译 |
| 7 | 编译/跑很慢 | 配置太大 | 先 `Nsample=64 / n_steps 小` 把 pipeline 跑通,再 scale |
| 8 | 指标缺列 | extractor 漏 signal | `compute_metrics` 静默 skip;检查输出含全部预期 key(docker extractor 测试已查) |

**Debug 的黄金顺序:** 先确认 env 单步 finite → 再确认 MDAC plan/solve finite → 再确认 extractor 出全 key → 再确认 AL 真活(mdac≠dial)→ 最后才看幅度/调参。

---

## 5. Phase B 接线 —— ✅ 已完成

所有组件已接 + docker 验证 active(`test_mdac_ablation_docker.py`),harness = `solvers/single/mdac/experiment.py::make_mdac(task, method)`(flag → env `stiffness_mode` + solver `geometry_fn`/`retraction`/`prior`/`aug`):

1. ✅ **geometry_fn(RQ3):** arm/humanoid env 的 `mdac_geometry_fn`(clean-state ∂(½‖C‖²)/∂U,纯 primitive 无 mjx;物理耦合的 surface/balance 走 AL)。
2. ✅ **CFS 回缩:** `make_mdac_retraction(env)` = genemetry `CfsRetraction(backend="jax", filter_fn=...)`,filter 在 `env.mdac_constraint` 上做线性化投影(像 2GO)。
3. ✅ **flag-driven stiffness(RQ2):** env `stiffness_mode`(none/log_spd/euclid/fixed)。
4. ✅ **耦合退火:** backend `_adaptive_mults`(rho↑/kappa↑,kappa 来自 genemetry `ScheduleOverlay`;sigma↓ 归 base schedule)。
5. ⬜ **RL prior 真训练(RQ4 唯一待办):** seam 已验证(`prior=` 生效),但要真数字得用 M6 `train_rl_prior` 在每个 env 上训 brax PPO,再把策略当 prior 传进 `make_mdac(..., prior=policy)`。

---

## 6. 下一步

1. **(可选先做)训 RL prior** —— M6 `train_rl_prior` 在 arm/humanoid 上各训一个 brax PPO,产出 RQ4 的真 prior。
2. **跑实验** —— 用 §1 的轻量脚本(`make_mdac` + 通用指标 + `aggregate_by`)跑 §2 的全矩阵,出主结果表。

> 第一次跑用小配置(Nsample=64, n_steps 小, seeds=4)把脚本/落盘/聚合跑通,再 scale。
