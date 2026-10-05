# Testing

## Fast local checks

```bash
python scripts/release/validate_release.py
pytest test/unit/test_core_constraints.py test/unit/test_experiment_config.py
python -m build
python -m twine check dist/*
```

## Test classes

| Class | Location | Default CI behavior |
| --- | --- | --- |
| Unit | `test/unit` and `test/algos` | Core subsets on Python 3.10/3.12 |
| Integration | `test/integration` | Selected portable runner checks |
| Simulator | marked `requires_mujoco`, `requires_brax`, or similar | Separate simulator job or automatic skip |
| Docker acceptance | `scripts/validation/docker` | Explicit invocation inside project image |
| Performance | `test/performance` | Scheduled/manual benchmark environment |
| GPU/Isaac | `gpu` or `isaac_lab` marker | Opt-in qualified host |

Optional capability markers skip when the dependency is unavailable. A file
that only exposes a `main()` acceptance probe belongs in `scripts/validation`,
because pytest will not execute it as a test.

## Docker probes

The scripts under `scripts/validation/docker` are scenario acceptance gates.
Run the command documented at the top of each file in the corresponding image.
They return a nonzero process status on failure.

## Adding tests

Test a stable contract or a failure mode. Avoid duplicating implementation
details. Mark external capabilities and keep imports lazy enough for collection
in a core environment.
