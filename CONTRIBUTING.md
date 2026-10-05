# Contributing to GenerativeDynamics

GenerativeDynamics accepts focused changes to planners, constraints,
environments, evaluation, deployment, documentation, and release tooling.

## Set up a contributor environment

```bash
git clone --recurse-submodules https://github.com/hhhhzl/enerdynamics.git
cd enerdynamics
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all,dev,docs]"
```

Use a feature branch. Keep generated results, checkpoints, caches, and local
datasets outside commits.

## Before opening a pull request

```bash
python scripts/release/validate_release.py
pytest test/unit test/algos -m "not performance and not gpu and not docker and not brax and not isaac_lab"
python -m build
python -m twine check dist/*
mkdocs build --strict
```

Run simulator or Docker acceptance probes when the change touches their paths.
Describe the environment and command in the pull request.

## Architecture rules

- Keep solver mathematics behind solver and method-plugin contracts.
- Keep task physics and state semantics inside environment/task adapters.
- Keep output layout, matrix expansion, and aggregation in the runner.
- Keep optional dependencies lazy and name the install extra in errors.
- Add configuration names to the release catalogue only when they are supported.
- Do not introduce soft-robot, co-design, or 3DGS systems into the V1 branch.

## Algorithm contributions

Include a minimal config, the mathematical reference, deterministic smoke
coverage, failure behavior, and benchmark metadata. Comparisons must share task,
seed, horizon, sampling budget, compilation policy, and hardware. Raw results
belong with the benchmark artifact, not as an unexplained README number.

## Pull request description

State the concrete trigger and resulting behavior. Include:

- the public or internal contract changed;
- installation or compatibility impact;
- validation commands and outcomes;
- benchmark protocol when performance is claimed;
- migration steps for configuration or result-schema changes.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
