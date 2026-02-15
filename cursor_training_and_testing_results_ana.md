# Training and testing results analysis for dpcc
_Exported on 2/14/2026 at 13:59:38 EST from Cursor (2.4.28)_

---

**User**

@enerdynamics/solvers/single/dpcc 我发现我训练4d的时候， @scripts/dpcc_train.py ，跑@scripts/dpcc_test_diffusion.py 得到不错的结果，但是跑9d后@scripts/dpcc_train_9d.py  ，跑@scripts/dpcc_test_diffusion.py 结果就不太对。帮我分析一下可能的原因

---

**Cursor**

我先把 4D/9D 的训练脚本、测试脚本，以及 `enerdynamics/solvers/single/dpcc` 里的实现并排读一遍，重点看：维度/归一化/网络输入输出 shape、数据集字段、config 默认值、以及 `dpcc_test_diffusion.py` 是否仍按 4D 假设在评估。然后我会给出最可能的 3-5 个原因和你可以立刻验证的检查点。

我已经看到 4D/9D 的训练脚本差异点：`dpcc_train_9d.py` 强制要求 `observation_dim==9` 且 `action_dim==7`，而测试脚本默认用 `env_name=d3il_avoiding_9d`、`dataset=avoiding-d3il-9d`，并且对 9D 情况会“强制”使用内置 indices（避免 YAML 里残留 4D 的索引）。下一步我会把 `dpcc_plan_torch.py`/`projector.py`/`avoiding_adapter.py` 里对 state/action 维度与约束的假设找出来，看是否仍隐含 4D 或某种坐标语义不一致。

这里已经露出一个很可疑的点：`AvoidingDPCCAdapter` 的注释和实现都假设环境 action 是 2 维的 \([dx,dy]\)，但 9D 训练脚本明确 action_dim=7（`qdot1..7`），测试脚本也默认跑 `d3il_avoiding_9d`。如果 9D 环境的 `env.step`/`robot_state` 语义不同，这个 adapter 可能在 reset/step 阶段把 action 维度或含义搞错，从而导致“训练似乎正常但 rollout 很怪”。我接下来会把 `D3ILAvoiding9DPlugin` 和对应 env 的 `reset/step/robot_state` 看一下，确认 action 预期是什么、obs 9D 的顺序是什么。

你这个现象**最像是“9D 的评估脚本参数/索引没对齐”，而不是训练本身突然坏掉**。在你现在这套代码里，`dpcc_test_diffusion.py` 和实验框架跑 9D 的方式其实不一样，导致 4D OK、9D 就“不太对”。

## 1) `dpcc_test_diffusion.py` 默认把 9D 的 `dt` 用错了（4D 恰好不受影响）
`DPCCBackendTorch` 里投影/动力学约束用的 \(dt\) 来自 `plan_config["dt"]`，否则回退到 `projection_eval.yaml` 的 `dt['avoiding']`：

```104:110:.../enerdynamics/solvers/single/dpcc/backends/dpcc_plan_torch.py
dt_cfg = self.constraint_config.get("dt")
...
dt = self.plan_config.get("dt", dt_default if dt_default is not None else 0.1)
delta_t = float(dt) * float(dt_multiplier)
```

而 `projection_eval.yaml` 里把 avoiding 的 `dt` 写成了 **1**（这是给 4D 的 delta-action 用的）：

```15:17:.../enerdynamics/solvers/single/dpcc/config/projection_eval.yaml
dt: {
  'avoiding': 1,    # Since a = [delta_x, delta_y] and not [vx, vy]
}
```

但是 9D 环境/数据是 **qdot 积分**，真实 `dt` 是 **0.035**（见 9D eval 配置）：

```4:22:.../configs/d3il_avoiding/dpcc_train_eval_9d.yaml
env_params:
  dt: 0.035
...
method_params:
  dt: 0.035
```

**关键差异**：实验框架的 `DPCCMethodPlugin` 会把 `dt` 明确塞进 `plan_config`（所以 9D 正常），但 `dpcc_test_diffusion.py` 的 CLI **没有 `--dt`**，因此 9D 会错误地沿用 `dt=1`，一旦你用 `dpcc-c / dpcc-t / gradient / tightened` 这类带 dynamics/projection 的 variant，结果就会明显跑偏。

## 2) `dpcc_test_diffusion.py` 对 9D “强制 indices”太简化，缺了 q 维度（会破坏/误配 dynamics 约束）
脚本里 9D 的默认 indices 只给了 `x,y` 和 `qdot1..7`，**没有 `q1..q7`**：

```47:60:.../scripts/dpcc_test_diffusion.py
if env_name == "d3il_avoiding_9d":
    return {
        "observations": {"x": 0, "y": 1},
        "actions": {"qdot1": 0, ... "qdot7": 6},
    }
```

并且它在 9D 时会**直接忽略 YAML 里更完整的 `avoiding_9d` 索引**：

```69:77:.../scripts/dpcc_test_diffusion.py
if env_name == "d3il_avoiding_9d":
    # Force 9D semantics ...
    return _default_indices(env_name)
```

但真正“框架内”的 DPCC 方法插件在 9D 会用 `avoiding_9d`（包含 `q1..q7`）：

```146:153:.../enerdynamics/experiments/plugins/methods/dpcc.py
index_key = f"{robot_name}_9d" if "9d" in exp else robot_name
obs_indices = obs_indices_map.get(index_key, ...)
act_indices = act_indices_map.get(index_key, ...)
indices = {"observations": obs_indices, "actions": act_indices}
```

如果 indices 不含 `q1..q7`，那 9D 的“q 与 qdot 一致性”dynamics 约束（代码里专门写了 9D 分支）就**无法正确启用**：

```53:66:.../enerdynamics/solvers/single/dpcc/patch/constraints_helpers.py
if "9d" in exp and action_dim >= 7
 and all(f"q{i}" in act_obs_indices ...)
 and all(f"qdot{i}" in act_obs_indices ...):
    dynamic_constraints = [("deriv", [q_i, qdot_i])...]
```

这会让你在 9D 下用 DPCC 投影/梯度时，约束系统和真实语义严重不一致，表现出来就是“rollout 结果怪/不稳定/违反约束/成功率很低”。

## 3) 训练侧也可能确实更难，但先把“评估对齐”排掉
9D 比 4D 难很多（维度更高、action 7 维、分布更复杂），同样的网络宽度/训练步数可能不够；不过你这里最明显的是**评估脚本没把 9D 的 dt/indices 对齐**，这属于“必修 bug”，优先级高于调参。

## 你可以怎么验证（不改代码也能快速定位）
- **用实验框架的 9D eval 跑同一个 checkpoint**：`configs/d3il_avoiding/dpcc_train_eval_9d.yaml`（它会带 `dt=0.035` 且 indices 正确）。如果这样结果明显变好，基本就坐实是 `dpcc_test_diffusion.py` 的对齐问题。
- 如果你坚持用 `dpcc_test_diffusion.py`，那至少要确保：
  - **variant 先用 `diffuser`**（完全关掉 projector/约束投影），看纯扩散 policy 是否合理；
  - 一旦要用 `dpcc-* / gradient`，就需要它支持 9D 的 `dt=0.035` 和完整 `avoiding_9d` indices（这两点在当前脚本里缺失/被覆盖）。

如果你把你实际运行 `dpcc_test_diffusion.py` 时的命令（variant/epoch/diffusion-loadpath）贴出来，我可以更精确地判断你现在到底踩中了上面哪一条（尤其是你用没用投影类 variant）。

---

**User**

目前训练用的什么样的网络？容量够用么？

---

**Cursor**

我先定位你 9D 训练用的模型定义（UNet1DTemporalCondModel）和对应超参（`dim`, `dim_mults`, `hidden_dim` 等），再对比 4D 的配置，看输入维度从 6→16 后容量是否显得偏小，然后给你一套“最小改动”的增容建议。

我已经确认 9D 训练用的是 `UNet1DTemporalCondModel`（1D temporal U-Net），base channel `dim=32`、层宽按 `(1,2,4,8)` 放大。接下来我把 4D 训练 config 找出来对比一下是否同样容量；同时结合 4D(6维 transition) vs 9D(16维 transition) 解释为什么 9D 更容易“容量不够/欠拟合”。

### 现在训练用的网络是什么？
你 9D 训练脚本 `scripts/dpcc_train_9d.py` 用的模型来自配置 `configs/d3il_avoiding/dpcc_diffusion_train.py`：

- **模型**：`models.UNet1DTemporalCondModel`
- **Diffusion**：`models.GaussianDiffusion`
- **关键超参**：`horizon=8`, `n_diffusion_steps=20`, `dim=32`, `dim_mults=(1,2,4,8)`, `condition_dropout=0.25`

对应配置在：

```15:33:/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/enerdynamics/configs/d3il_avoiding/dpcc_diffusion_train.py
"model": "models.UNet1DTemporalCondModel",
...
"horizon": 8,
"n_diffusion_steps": 20,
...
"dim": 32,
"dim_mults": (1, 2, 4, 8),
"condition_dropout": 0.25,
```

这个 `UNet1DTemporalCondModel` 是**沿时间维做 1D Conv 的 U-Net**（多层 ResidualTemporalBlock + 下采样/上采样），定义在：

```84:168:/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/enerdynamics/third_party/diffuser/models/unet1d_temporal_cond.py
class UNet1DTemporalCondModel...
    dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
```

在 9D 下 `transition_dim = observation_dim + action_dim = 9 + 7 = 16`（训练脚本里就是这么传的），所以通道大概是 **16 → 32 → 64 → 128 → 256**（再对称上采样回去）。

### 容量够用么？
- **从“纯参数规模”看：不算小，但对 9D 更容易不够用。**9D 的建模复杂度比 4D 高很多（状态/动作维度从 \(4+2=6\) 变成 \(9+7=16\)，而且 tcp\_xy 与 q/qdot 的耦合更强），同样的 `dim=32` 往往更容易出现欠拟合/采样质量差。
- **你 config 里写的 `hidden_dim` / `attention` 目前基本不生效**：`dpcc_train_9d.py` 构造模型时只传了 `dim`、`dim_mults`、`condition_dropout` 等；`hidden_dim=256`、`attention=False` 虽然在 config 里，但没有被传进 `UNet1DTemporalCondModel`（它本身也没有 `attention` 参数）。所以你当前“容量旋钮”主要就是 **`dim` 和 `dim_mults`**。

### 建议你怎么判断是不是容量问题（最快）
- **看训练 vs 测试 loss**：如果 9D 的 train/test loss 都明显高、下降慢或早早平台化，更像容量/训练步数不够；如果 train loss 很低但 rollout 差，更像评估/约束对齐问题（你前面那个 9D `dt/indices` 就属于这类）。
- **先用 `variant=diffuser` 跑 9D rollout**（关掉投影/约束），单看扩散策略本身；如果 diffuser 都很差，才更像“模型/训练不够”。

### 如果确实要“加容量”，最有效的改动是什么？
- **把 `dim: 32 → 64`（甚至 128）**通常是对 9D 最直接有效的增容方式；其次才是加深 `dim_mults`（但 `horizon=8` 下 U-Net 深到一定程度收益会变小）。
- **训练步数也可能要加**：9D 分布更复杂，同样 1e5 steps 不一定够。

