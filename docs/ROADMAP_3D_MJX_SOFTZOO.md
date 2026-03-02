# 3D / MJX / SoftZoo 接入路线图

> 基于现有 TaskSpec 抽象，分阶段引入 3D、高维 MJX、SoftZoo 软体机器人支持。

---

## 一、现状与前提

### 1.1 已完成的基础设施

- **TaskSpec**：`extract_position(state)`、`position_dim`、`success_criterion` 已抽象
- **Solver 全量接入**：ebmbd、mbd、mdoc、dpcc、safediffuser 均已使用 `position_extractor` / `position_dim`
- **EnvironmentPlugin**：`extract_position`、`get_state_dim`、`create_energy` 等接口已稳定
- **已有 3D 环境**：`drone_full_3d`、`drone_full_3d_mujoco`、`drone_full_3d_isaac` 已存在，但尚未通过 TaskSpec 统一

### 1.2 需要扩展的点

| 维度 | 当前 | 目标 |
|------|------|------|
| position_dim | 固定 2 | 支持 3（及未来更高维） |
| get_default_task_spec | 仅用 env_plugin，env_name 未用 | 按 env_name 返回 2D/3D/高维 spec |
| success_criterion | 默认 2D 点目标、D3IL 线目标 | 3D 点/面目标，SoftZoo 自定义 |
| 障碍物 SDF/CBF | 2D 圆/多边形 | 3D 球/椭球/凸多面体 |
| 物理后端 | MuJoCo (CPU)、Isaac | 增加 MJX (JAX/GPU) |
| 软体状态 | 无 | SoftZoo 高维可变形状态 |

---

## 二、Phase 1：3D 支持（最小改动）

**目标**：让现有 3D 环境（如 drone_full_3d）通过 TaskSpec 正确运行，无需改 solver 逻辑。

### 2.1 TaskSpec 扩展

```
genedynamics/core/task_spec.py
├── Legacy3DTaskSpec(TaskSpec)
│   ├── extract_position(state) → state[:3] 或 state[3:6]（取决于 state layout）
│   └── position_dim = 3
├── 修改 get_default_task_spec(env_plugin, env_name):
│   └── 若 env_name 匹配 "drone_full_3d*" / "3d" → Legacy3DTaskSpec
│   └── 否则 → 保持现有逻辑
└── 修改 TaskSpec.success_criterion：
    └── 对 3D：支持 ||final_pos - target|| < margin（点目标）
```

### 2.2 EnvironmentPlugin 扩展

- `EnvPluginTaskSpecAdapter`：增加 `position_dim`，从 `env_plugin.get_position_dim()` 或 `env_name` 推断
- 为 `DroneFull3DPlugin` 等增加 `get_position_dim() -> 3`（可选，用于 EnvPluginTaskSpecAdapter）

### 2.3 障碍物与约束

- **obstacle_sdf.py / cbf.py**：已有 `position_extractor`，只要 `position_dim=3` 时传入的 pos 是 (x,y,z) 即可
- **3D 障碍物**：`genedynamics/envs/obstacles/` 已有 `convex.py`、`nonconvex.py`，需确认 SDF 在 3D 下的正确性
- **CFS / 约束管线**：沿用 `position_extractor`，无需改接口

### 2.4 可视化

- `diffusion_3d.py`、`trajectory_3d.py` 已用 `env_plugin.extract_position`，3D 环境只需正确实现 `extract_position` 返回 (x,y,z)

### 2.5 交付物

- [ ] `Legacy3DTaskSpec` + `get_default_task_spec` 分支
- [ ] Drone 3D 环境在 experiment 框架下跑通
- [ ] 障碍物 3D SDF 验证

---

## 三、Phase 2：MJX（高维 MuJoCo X）支持

**目标**：引入 `mujoco-mjx` 作为 JAX 物理后端，支持批量仿真与可微仿真。

### 3.1 MJX 简介

- **mujoco-mjx**：MuJoCo 的 JAX 实现，支持 GPU/TPU、批处理、`jax.grad` 可微
- **适用场景**：MBD/CEM/EBM 等需大量 rollout 的 solver，可批量并行；梯度可用于轨迹优化

### 3.2 架构设计

```
genedynamics/core/backends/
├── mujoco_adapter.py      # 现有 CPU MuJoCo
└── mjx_adapter.py         # 新增：MJX 物理后端
    ├── put_model / put_data（MJX 要求）
    ├── step_batch(states, actions) → next_states  # 批处理
    └── 与 PhysicsBackend 接口对齐
```

### 3.3 环境层

```
genedynamics/envs/
├── drone_full_3d_mujoco.py    # 现有
└── drone_full_3d_mjx.py       # 新增
    └── physics_backend = 'mjx'
    └── 使用 MJxModel / MjxData，state 与 MuJoCo 一致
```

### 3.4 实验框架

- `experiment.py`：`physics_backend in ['mujoco', 'isaac', 'mjx']` 时处理障碍物等
- `DroneFull3DPlugin`：支持 `physics_backend='mjx'` 分支

### 3.5 Solver 利用 MJX 的可微性（可选）

- MBD/EBM 的 JAX backend 已支持 JAX 环境；若 rollout 用 MJX，可自然获得 `jax.grad`
- 可为 trajectory optimization 增加「可微物理」模式，对初始状态/参数求梯度

### 3.6 交付物

- [ ] `MjxPhysicsBackend` 实现
- [ ] `DroneFull3DMjxEnv` 或 `physics_backend='mjx'` 分支
- [ ] 在 config 中支持 `physics_backend: mjx`
- [ ] （可选）可微 rollout 示例

---

## 四、Phase 3：SoftZoo 软体机器人

**目标**：接入 SoftZoo，支持软体机器人形态-控制联合优化与避障规划。

### 4.1 SoftZoo 简介

- **仓库**：mitibmwatsonailab/softzoo（GitHub）
- **能力**：可微软体仿真、多环境（沙漠、湿地、冰面等）、形态+控制联合优化
- **状态**：高维（网格/控制点变形），需定义「控制点」或「质心」作为 TaskSpec 的 position

### 4.2 状态与 TaskSpec

软体状态通常为：

- 网格顶点坐标、速度，或
- 控制点 / 关键点坐标

**建议**：

- 定义 `SoftZooTaskSpec(TaskSpec)`：
  - `extract_position(state)`：取质心或指定控制点 (x,y) 或 (x,y,z)
  - `position_dim`：2（平面）或 3（3D）
  - `success_criterion`：按任务自定义（如质心到达目标区域）

### 4.3 环境插件

```
genedynamics/experiments/plugins/environments/
└── softzoo.py
    └── SoftZooEnvironmentPlugin(EnvironmentPlugin)
        ├── create_env() → SoftZoo gym-like env
        ├── create_energy() → 基于质心/控制点与目标的距离
        ├── get_state_dim() → 高维（e.g. 数百维）
        └── extract_position() → 质心或控制点子集
```

### 4.4 与 genedynamics 的集成方式

**方案 A：Adapter 包装**

- SoftZoo 提供 gym-like 接口
- `SoftZooEnvAdapter` 包装，实现 `step`、`reset`、`state`、`target`
- 通过 `UnifiedEnvAdapter` 接入现有管线

**方案 B：直接实现 EnvironmentPlugin**

- 在 `create_env` 中构造 SoftZoo 环境
- 通过 plugin 暴露 `extract_position`、`get_state_dim` 等

### 4.5 Solver 适配

- **高维 state**：MBD/EBM/CEM 等已支持可变 `state_dim`，只要 `position_extractor` 返回低维 position 即可
- **DPCC/SafeDiffuser**：依赖 4D/9D 假设，需评估是否新增「SoftZoo 模式」或保持为独立实验线

### 4.6 障碍物与软体碰撞

- SoftZoo 自带地形/障碍；genedynamics 障碍物主要用于规划层
- 若规划在「降维空间」（如质心）进行，可用现有 2D/3D 障碍物
- 精细碰撞需在 SoftZoo 内部或扩展其接口

### 4.7 交付物

- [ ] `SoftZooTaskSpec`
- [ ] `SoftZooEnvironmentPlugin`
- [ ] SoftZoo env 在 experiment 框架下可配置、可运行
- [ ] 至少一个 solver（如 MBD/EBM）在 SoftZoo 上验证

---

## 五、依赖与风险

| 组件 | 依赖 | 风险 |
|------|------|------|
| 3D | 无新增 | 低 |
| MJX | mujoco>=3.0, mujoco-mjx | MJX API 可能随版本变化 |
| SoftZoo | softzoo 安装、PyTorch/JAX 版本 | 与 genedynamics 的 JAX 版本需兼容 |

---

## 六、推荐实施顺序

1. **Phase 1（3D）**：改动小，验证 TaskSpec 在 3D 下的完整性
2. **Phase 2（MJX）**：利用现有 JAX 与 MuJoCo 基础，提升仿真效率
3. **Phase 3（SoftZoo）**：独立性强，可与前两阶段并行探索

---

## 七、配置示例（目标形态）

```yaml
# 3D drone + MJX
env_name: drone_full_3d
env_params:
  physics_backend: mjx  # 或 mujoco, isaac

# SoftZoo
env_name: softzoo_flat
env_params:
  robot_type: caterpillar
  terrain: flat
  position_mode: centroid  # 或 keypoints
```

---

*文档版本：2025-03*
