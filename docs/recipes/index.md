# Recipes

Recipes are published configurations with a stable task and output contract.

| Goal | Start with | Required capability |
| --- | --- | --- |
| Learn the runner | `configs/single_2d/mbd.yaml` | Core |
| Compare constrained diffusion | `configs/single_2d/mdcoas.yaml` | Core + optimization |
| Run D3IL avoidance | `configs/d3il_avoiding/mbd.yaml` | D3IL |
| Try a learned D3IL planner | `configs/d3il_avoiding/dpcc.yaml` | D3IL + Torch |
| Plan quadruped footholds | `configs/quadruped/stepping_stones_2d/main/twogo.yaml` | Simulation + optimization |
| Plan a humanoid corridor | `configs/humanoid/corridor_2d/main/twogo_zone_a.yaml` | Core for planning; deploy extras for execution |
| Compare peg insertion controllers | `configs/arm/peg_insert/baseline/dial.yaml` | Simulation |
| Run MGA peg insertion | `configs/arm/peg_insert/main/mga.yaml` | Simulation; frozen learned assets for formal runs |
| Run compliant surface scan | `configs/arm/surface_scan/main/mga.yaml` | Simulation |

Always begin with `--dry-run`, then use a development root and one seed. Scale
to the formal matrix only after inspecting saved status, metrics, and trajectory
artifacts.

Start with the [planar CPU smoke](planar-cpu.md), whose CLI output and artifact
contract are pinned for V1. The [humanoid corridor recipe](humanoid-corridor.md)
shows the same workflow on a robot planning task.
