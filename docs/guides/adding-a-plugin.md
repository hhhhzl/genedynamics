# Add a plugin

The experiment registry accepts five plugin types: method, environment, metric,
visualization, and obstacle generator.

## Method example

```python
from genedynamics.experiments.framework import MethodPlugin


class MyPlannerPlugin(MethodPlugin):
    @property
    def name(self):
        return "my_planner"

    def create_planner(self, env, energy, config):
        return MyPlanner(env=env, energy=energy, **config)

    def plan(self, planner, initial_state, rng):
        result = planner.plan(initial_state, rng)
        return {
            "states": result.states,
            "actions": result.actions,
            "initial_state": initial_state,
        }
```

Register the instance in `register_all_plugins`, add the method name to the V1
catalogue only when it is intended for release, and create one recipe that
proves the complete path.

## Required checks

1. Keep optional dependencies lazy and raise an install command that names the
   correct extra.
2. Return JSON-serializable result metadata and arrays with stable dimensions.
3. Accept the seed/random key supplied by the runner.
4. Add a unit test for the adapter boundary and an integration smoke test for
   the runner path.
5. Record device, compile/warm-up policy, and task dimensions in benchmarks.
6. Update configuration validation, compatibility, and user docs.

Environment plugins additionally define environment creation, energy creation,
state dimension, position extraction, and optional task-owned reset/execution
environment behavior. Metric plugins must keep scalar outputs serializable and
may expose separate artifacts through `pop_artifacts()`.
