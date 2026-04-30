# 注册与接线清单 (Registration & Wiring Checklist)

本文档列出 genedynamics 中所有方法、环境、插件的注册点及接线关系，便于扩展与排查。

---

## 1. 实验框架 (Experiment Runner)

### 1.1 入口

```
python -m genedynamics.experiments.runner <config.yaml>
```

- **模块**: `genedynamics/experiments/runner.py`
- **插件注册**: `register_all_plugins(runner)` 在 `run_all()` 前调用

### 1.2 方法插件 (Method Plugins)

| 名称 | 插件类 | 注册键 | 配置文件 method |
|------|--------|--------|------------------|
| MBD | MBDMethodPlugin | `method` | `mbd` |
| EB-MBD | EBMBDMethodPlugin | `method` | `ebmbd` |
| MDOC | MDOCMethodPlugin | `method` | `mdoc` |
| CFS-MBD | CFSMBDMethodPlugin | `method` | `cfsmbd` |
| CFS-MBD Full (MD-COAS) | CFSMBDFullMethodPlugin | `method` | `cfsmbd_full` |
| MBD3D | MBD3DMethodPlugin | `method` | `mbd3d` |
| **MRMFMBD** | **MRMFMBDMethodPlugin** | **`method`** | **`mrmfmbd`** |
| D3IL Unified | D3ILUnifiedMethodPlugin | `method` | `d3il_unified` |
| 2GO | TwoGOMethodPlugin | `method` | `twogo` |
| DPCC | DPCCMethodPlugin | `method` | `dpcc` |
| SafeDiffuser | SafeDiffuserMethodPlugin | `method` | `safediffuser` |

**注册代码** (`runner.py`):

```python
runner.register_plugin(MRMFMBDMethodPlugin(), 'method')
```

### 1.3 环境插件 (Environment Plugins)

| 名称 | 插件类 | 注册键 | 配置文件 env_name |
|------|--------|--------|-------------------|
| Single Integrator 2D | SingleIntegrator2DPlugin | `environment` | `single_integrator_box_2d` |
| Double Integrator 2D | DoubleIntegrator2DPlugin | `environment` | `double_integrator_box_2d` |
| Quadruped Flat (MJX) | QuadrupedFlatMjxPlugin | `environment` | `quadruped_flat_mjx` |
| Quadruped Go2 (MJX) | QuadrupedGo2MjxPlugin | `environment` | `quadruped_go2_mjx` |
| Humanoid Simplified | HumanoidSimplifiedMjxPlugin | `environment` | `humanoid_simplified_mjx` |
| Humanoid G1 | HumanoidG1MjxPlugin | `environment` | `humanoid_g1_mjx` |
| Drone Full 3D | DroneFull3DPlugin | `environment` | `drone_full_3d_mjx` |
| Mujoco Scene Mapping | MujocoSceneMappingPlugin | `environment` | `mujoco_scene_mapping` |
| D3IL Avoiding | D3ILAvoidingPlugin | `environment` | `d3il_avoiding` |
| ... | ... | ... | ... |

**MRMFMBD 软体协同设计**: 通过 `task_domain: jax_mpm` 走 baseline_platform。

### 1.4 插件导出链

```
genedynamics/experiments/plugins/__init__.py
  └── from .methods import MRMFMBDMethodPlugin
        └── genedynamics/experiments/plugins/methods/__init__.py
              └── from .mrmfmbd import MRMFMBDMethodPlugin
                    └── genedynamics/experiments/plugins/methods/mrmfmbd.py
```

---

## 2. MRMFMBD 接线

### 2.1 创建 Planner 流程

```
ExperimentRunner.run_single()
  → method_plugin.create_planner(env, energy, method_config)
```

**MRMFMBDMethodPlugin.create_planner** 接线:

| 输入 | 来源 | 用途 |
|------|------|------|
| `env` | env_plugin.create_env() | environment instance |
| `energy` | env_plugin.create_energy() | LegacyEnergyFunctional |
| `config` | method_params + scheduler + obstacles | 扩散参数、fidelity 等 |

**输出**: `MRMFMBDSolver`

### 2.2 MRMFMBDSolver 内部接线

```
MRMFMBDSolver
  ├── dynamics: EnvDynamicsAdapter(env)
  ├── energy: LegacyEnergyFunctional
  ├── backend: RuntimeBackendManager.get_backend()  # jax
  ├── fidelity_simulator: EnvFidelitySimulator(env, energy, position_extractor, ...)
  ├── fidelity_ladder: FidelityLadder(FidelityConfig, K=Ndiffuse)
  └── _backend_impl: MRMFMBDBackendJax(...)
```

### 2.3 Env 协议要求

- `jax_transition(state, action)` — 必需，用于 JAX rollout
- `jax_transition_fidelity(state, action, level)` — 可选，multi-fidelity (mode + fidelity systems)

### 2.4 TaskSpec

- `get_default_task_spec(env_plugin, env_name)` → `extract_position`, `position_dim`

---

## 3. Deploy 管道 (Pipeline)

### 3.1 适用场景

Deploy 管道用于 **quadruped / humanoid / uav3d** 等机器人仿真与部署。

**MRMFMBD 不接入 Deploy 管道**：MRMFMBD 面向软体机器人协同设计，通过 baseline_platform `configs/soft_robot/*/*.yaml` 运行。

### 3.2 Deploy 支持的 Planner

| planner_type | 工厂函数 | 适用 |
|--------------|----------|------|
| stand | QuadrupedStandPlanner | 零动作 |
| mbd | make_mbd_planner | MBD 扩散 |
| cfsmbd | make_cfsmbd_planner | CFS-MBD 每步 |
| cfsmbd_full | make_cfsmbd_full_planner | MD-COAS |

---

## 4. 核心 Registry

| Registry | 模块 | 用途 |
|----------|------|------|
| 实验插件 | runner.register_plugin | method / environment / metric / visualization |
| 环境 | envs.factories.make_env | env_name → EnvClass |
| 能量 | envs.factories.make_energy | env_name → EnergyFunctional |
| Backend | core.backends.get_backend | jax / numpy |
| 约束 | core.constraints.core.registry | convexifier, solver, operator |
| 机器人 | robots.get_robot_registry | quadruped, humanoid |
| Deploy Profile | deploy.profiles | robot_type, model_id → profile |

---

## 5. 扩展新方法检查清单

1. **方法插件**: 继承 `MethodPlugin`，实现 `name`, `create_planner`, `plan`
2. **注册**: 在 `runner.py` 的 `register_all_plugins` 中 `runner.register_plugin(XxxMethodPlugin(), 'method')`
3. **导出**: 在 `plugins/methods/__init__.py` 和 `plugins/__init__.py` 中 export
4. **配置**: 在 `configs/` 下创建 YAML，`method: <name>`
5. **脚本**: 在 `scripts/` 下创建 `run_xxx.sh`（可选）

---

## 6. 扩展新环境检查清单

1. **环境插件**: 继承 `EnvironmentPlugin`，实现 `name`, `create_env`, `create_energy`
2. **注册**: 在 `register_all_plugins` 中 `runner.register_plugin(XxxEnvPlugin(), 'environment')`
3. **导出**: 在 `plugins/environments/__init__.py` 和 `plugins/__init__.py` 中 export
4. **TaskSpec**: 若需自定义 position 提取，在 `core.task_spec` 中扩展
