# `deploy/observers/`

Observers are pure side-effect sinks. The runner forwards every step's
`(state, intent, cmd, info)` tuple to every registered observer; observers
never feed back into controller behavior. Failure inside an observer is
caught and logged — it never aborts the episode.

## Files

| File | Purpose | Output |
|---|---|---|
| [`base.py`](base.py) | abstract base — every hook is a no-op by default | — |
| [`logger.py`](logger.py) | per-step JSONL logger + `summary.json` | `<out_dir>/<episode_id>.jsonl` |
| [`recorder.py`](recorder.py) | in-memory state/action timeseries flushed on episode end | `<out_dir>/<episode_id>.npz` |
| [`mujoco_render.py`](mujoco_render.py) | legacy mp4/gif renderer | `<out_dir>/episodes/<episode_id>/render.mp4` |
| [`web_viz.py`](web_viz.py) | legacy Flask viewer (consumed by `cli.py --viz`) | served at `localhost:<port>` |

## Observer protocol

```python
class Observer(Protocol):
    name: str
    def on_episode_start(self, episode_id, metadata): ...
    def on_step(self, t, state, intent, cmd, info): ...
    def on_episode_end(self, summary): ...
```

The runner calls these in fixed order at episode start, every step, and
episode end. Multiple observers stack — preset entry order is preserved:

```python
observers = [
    {"registry_key": "observer.logger",   "out_dir": "results/run/"},
    {"registry_key": "observer.recorder", "out_dir": "results/run/"},
]
```

## Adding an observer

1. Subclass [`BaseObserver`](base.py) and override the hooks you care about.
2. Register under `observer.<name>` in [`registries.py`](../registries.py).
3. Add a dict entry to your preset's `observers` list.

For long rollouts on memory-constrained machines, set `RecorderObserver(include_state=False)`
— it drops `qpos`/`qvel` capture but keeps commands and metrics.

## Tests

- 6 unit tests in [`test_observers.py`](../../../test/unit/deploy/test_observers.py)
  cover JSONL formatting, summary file emission, NPZ shapes, padding for
  variable-width commands, and skip-state mode.
