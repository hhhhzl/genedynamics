# Humanoid corridor planning

The Zone A recipe exercises the 2GO constrained planner on a multimodal
corridor. V1 supports the planning configuration; humanoid walking and
push-to-line execution remain deferred.

## Inspect the resolved robot task

```bash
genedynamics-run \
  configs/humanoid/corridor_2d/main/twogo_zone_a.yaml \
  --device cpu --seed 0 --level 0 --dry-run
```

The command must exit with status `0` and identify:

```text
Configuration is valid (dry run mode)
Environment: humanoid_corridor_2d
Method: 2go
Levels: [0]
Seeds: [0]
```

## Run one development case

Install the optimization extra, then write results outside the formal matrix:

```bash
python -m pip install -e ".[optimization]"
genedynamics-run \
  configs/humanoid/corridor_2d/main/twogo_zone_a.yaml \
  --device cpu --seed 0 --level 0 \
  --development-root results/_development/v1-corridor
```

A complete run records the resolved configuration, the selected trajectory,
constraint metrics, and execution status. The planner may return different
valid corridor modes across platforms because floating-point reductions can
break symmetric ties; the output schema and safety checks are the stable V1
contract.
