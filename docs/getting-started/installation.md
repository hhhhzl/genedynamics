# Installation

## Supported base environment

- Python 3.10, 3.11, or 3.12
- Linux or macOS for the core package
- JAX CPU for the portable default

Create an isolated environment and install the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -c "import genedynamics; print(genedynamics.__version__)"
```

## Capability extras

| Extra | Adds | Use it for |
| --- | --- | --- |
| `optimization` | OSQP, CVXOPT, JAXopt, QPAX | QP-backed constraint paths and comparisons |
| `simulation` | MuJoCo/MJX, Brax, Gymnasium, PyBullet, trimesh | Robot simulation tasks |
| `manipulator` | Pinocchio, JAX kinematics and mesh proximity support | Serial-arm tools |
| `d3il` | D3IL Python/build dependencies | D3IL avoiding recipes |
| `torch` | Torch, Diffusers, QPTH | DPCC and SafeDiffuser integrations |
| `reports` | Jinja, pandas, seaborn | HTML and aggregate reports |
| `deploy` | FlatBuffers, WebSockets, OSQP | Runtime and AR transport paths |
| `docs` | MkDocs Material | Local documentation site |
| `dev` | Test, build, formatting, and release tools | Contributors |

Example:

```bash
python -m pip install -e ".[simulation,optimization,reports]"
```

## GPU JAX

GPU installation depends on the host driver and platform, so it is not a core
wheel dependency. On a compatible Linux/NVIDIA host:

```bash
python -m pip install -r requirements/gpu-jax.txt
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

The first V1 release does not qualify MGA GPU behavior. A visible GPU device
only proves installation; it does not prove numerical parity or performance.

## D3IL

D3IL is kept under `third_party/environments/d3il`. Clone with submodules and
run the integration setup:

```bash
git submodule update --init --recursive
./scripts/setup/setup_d3il.sh
python -m pip install -e ".[d3il,torch]"
```

See [the D3IL integration guide](../integrations/d3il.md) for dataset and
controller details.

## Verify a wheel

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
python -m pip install --force-reinstall dist/genedynamics-*.whl
python -c "import genedynamics; print(genedynamics.__version__)"
```

The release CI also inspects wheel contents to ensure required robot assets are
present and deferred V1 systems are absent.
