# SoftZoo Integration (enerdynamics)

This directory contains the vendored [SoftZoo](https://github.com/zswang666/softzoo) repository.

## Layout

- `softzoo/` - Python package (configs, engine, envs, tools)
- `algorithms/` - Designer, controller, diffsim, RL
- `scripts/` - Original SoftZoo scripts

## Assets

**Downloadable assets** go in `data/softzoo/assets/` (project root).

Configs stay here in `softzoo/configs/`.

See `docs/softzoo_assets.md` for download instructions.

## Integration Entry

- `genedynamics.envs.external.softzoo` - Bootstrap, schemas, adapters
- `scripts/third_party/check_softzoo.py` - Health check
- `scripts/third_party/setup_softzoo_assets.sh` - Asset setup

## Path Resolution

1. `SOFTZOO_ROOT` / default: `third_party/environments/softzoo`
2. `SOFTZOO_ASSETS_ROOT` / default: `data/softzoo/assets`
3. Configs: `softzoo/configs/env_configs/`
