"""Independent unified-runner adapters for receding contact controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from genedynamics.core.types import Trajectory
from genedynamics.experiments.framework.base import MethodPlugin
from genedynamics.solvers.single.mdac.core.method_registry import (
    METHOD_TABLE,
    resolve_method,
)
from genedynamics.solvers.single.mdac.experiment import make_controller


_FACTORY_KEYS = {
    "aug_lambda",
    "aug_rho",
    "policy_ckpt",
    "atacom_policy_ckpt",
    "reliability_ckpt",
}


@dataclass
class _ContactPlanner:
    solver: Any
    env: Any
    execution_env: Any
    task: str
    controller_method: str
    n_steps: int
    seed: int
    component_contract: Dict[str, Any]


class RecedingContactMethodPlugin(MethodPlugin):
    """Adapter that delegates algorithm semantics to the existing factory."""

    def __init__(self, plugin_name: str, default_controller_method: str) -> None:
        self._name = str(plugin_name)
        self._default_controller_method = str(default_controller_method)

    @property
    def name(self) -> str:
        return self._name

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> Any:
        del energy
        task = str(config.get("task") or getattr(env, "_experiment_task", ""))
        if not task:
            raise ValueError(f"method '{self.name}' requires a contact task")
        method = str(config.get("controller_method", self._default_controller_method))
        if self.name == "full_mdac":
            if method not in METHOD_TABLE:
                raise ValueError("full_mdac must resolve to an MDAC method contract")
            flags = resolve_method(method)
            if not flags.use_rl_prior:
                raise ValueError("full_mdac controller contract disables the RL prior")
            missing = [
                key for key in ("policy_ckpt", "reliability_ckpt")
                if not config.get(key)
            ]
            if missing:
                raise ValueError(
                    "full_mdac requires frozen learned components: "
                    + ", ".join(missing)
                )
        elif method in METHOD_TABLE and self.name in {
            "mppi", "pegasusflow", "issa", "atacom", "standalone_rl"
        }:
            raise ValueError(
                f"baseline plugin '{self.name}' cannot route through MDAC method '{method}'"
            )

        factory_kwargs = {
            key: config[key] for key in _FACTORY_KEYS if key in config
        }
        solver_cfg = {
            key: value for key, value in config.items()
            if key not in _FACTORY_KEYS
            and key not in {
                "constraint_pipeline", "scheduler", "obstacles",
                "obstacle_config", "env_plugin", "env_name", "task",
                "n_steps", "controller_method", "execution_env_params",
                "env_params", "suite", "task_level", "np_random_seed",
            }
        }
        execution_env = getattr(env, "_experiment_execution_env", None)
        model_env, solver = make_controller(
            task,
            method,
            level=config.get("task_level"),
            surface_seed=int(config.get("np_random_seed", 0)),
            prior=None,
            model_env=env,
            execution_env=execution_env,
            **factory_kwargs,
            **solver_cfg,
        )
        if model_env is not env:
            raise RuntimeError("contact controller silently replaced the model environment")
        contract: Dict[str, Any] = {
            "plugin": self.name,
            "controller_method": method,
            "policy_ckpt": config.get("policy_ckpt"),
            "atacom_policy_ckpt": config.get("atacom_policy_ckpt"),
            "reliability_ckpt": config.get("reliability_ckpt"),
        }
        if method in METHOD_TABLE:
            contract["mdac_flags"] = asdict(resolve_method(method))
        return _ContactPlanner(
            solver=solver,
            env=env,
            execution_env=execution_env or env,
            task=task,
            controller_method=method,
            n_steps=int(config.get("n_steps", 1)),
            seed=int(config.get("np_random_seed", 0)),
            component_contract=contract,
        )

    def plan(self, planner: _ContactPlanner, initial_state: Any, rng: Any) -> Dict[str, Any]:
        del rng
        import jax

        result = planner.solver.run_receding(
            initial_state,
            planner.n_steps,
            jax.random.PRNGKey(1000 + planner.seed),
            collect_states=True,
            synchronize_steps=True,
        )
        states = list(result.states)
        actions = list(result.actions)
        infos = list(getattr(result, "infos", ()))
        costs = list(getattr(result, "costs", ()))
        trajectory = Trajectory(states=states, actions=actions, info={
            "infos": infos,
            "costs": costs,
        })
        return {
            "trajectory": trajectory,
            "states": states,
            "actions": actions,
            "infos": infos,
            "costs": costs,
            "initial_state": initial_state,
            "execution_env": planner.execution_env,
            "receding_result": result,
            "component_contract": planner.component_contract,
        }


class FullMDACMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("full_mdac", "mdac")


class ModelBasedOnlyMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("model_based_only", "mdac_controllable")


class DIALContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("dial", "dial")


class MPPIContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("mppi", "mppi")


class PegasusFlowContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("pegasusflow", "pegasusflow")


class ISSAContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("issa", "issa")


class ATACOMContactMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("atacom", "atacom")


class StandaloneRLMethodPlugin(RecedingContactMethodPlugin):
    def __init__(self) -> None:
        super().__init__("standalone_rl", "rl")


__all__ = [
    "RecedingContactMethodPlugin",
    "FullMDACMethodPlugin",
    "ModelBasedOnlyMethodPlugin",
    "DIALContactMethodPlugin",
    "MPPIContactMethodPlugin",
    "PegasusFlowContactMethodPlugin",
    "ISSAContactMethodPlugin",
    "ATACOMContactMethodPlugin",
    "StandaloneRLMethodPlugin",
]
