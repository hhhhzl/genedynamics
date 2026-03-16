# Robot Registry (Multi-Backend)

Unified registry for quadruped, humanoid, and other robot models with multi-backend support.

## Adding a Quadruped

```python
from genedynamics.robots import register_quadruped

def _get_go2_mjcf_path() -> str:
    # Load from mujoco_menagerie or MUJOCO_MENAGERIE_PATH
    return "/path/to/go2.xml"

register_quadruped(
    model_id="go2",
    env_factory_name="quadruped_go2_physics",  # Register in envs/factories.py
    nq=19,
    nv=18,
    act_dim=12,
    model_path_resolver=_get_go2_mjcf_path,
    description="Unitree Go2",
)
```

Then add to `genedynamics/envs/factories.py`:

```python
elif name == "quadruped_go2_physics":
    from genedynamics.envs.quadruped_go2_physics import QuadrupedGo2PhysicsEnv
    return QuadrupedGo2PhysicsEnv(**kwargs)
```

Run: `genedynamics-deploy --robot quadruped --model go2`

## Adding a Humanoid

```python
from genedynamics.robots import register_humanoid

register_humanoid(
    model_id="h1",
    env_factory_name="humanoid_h1_physics",
    nq=34,
    nv=33,
    act_dim=19,
    model_path_resolver=lambda: "...",
    description="Unitree H1",
)
```

Run: `genedynamics-deploy --robot humanoid --model h1`

## Built-in Models

| robot_type | model_id | env_factory | dependency |
|------------|----------|-------------|------------|
| quadruped  | ant, flat, rough, push | quadruped_*_physics | gymnasium |
| quadruped  | go2      | quadruped_go2_physics | mujoco_menagerie |
| humanoid   | humanoid | humanoid_simplified_physics | gymnasium |
| humanoid   | h1       | humanoid_simplified_physics | dial-mpc |
| humanoid   | g1       | humanoid_g1_physics | mujoco_menagerie |

**Go2 / G1**: Clone mujoco_menagerie and set `MUJOCO_MENAGERIE_PATH`, or `pip install mujoco-menagerie` (if available)
