# ICDT: Cold-Start Handoff Without Transferring Experiment Artifacts

Updated: 2026-11-18 (America/New_York). Target: RSS 2027.

Continue from the current research findings without transferring the old 117 GiB `results/`.
This plan preserves the latest working tree, task definitions, manuscript, baseline source,
robot assets, environment versions, and research knowledge. Regenerate training data,
source banks, response models, actors, qualification reports, and formal evaluations.
Keep the old results on the source device; do not delete them.

This saves transfer space, but cannot restore the original checkpoints' training state.
Small summaries cannot reconstruct old confidence intervals or support episode re-audits.
Historical numbers inform the next design; new paper evidence requires accessible raw
artifacts or new runs. If the old device's data become inaccessible, the historical
summaries alone are insufficient to reproduce the experiments.

## File Guide

- `research_state.json`: machine-readable state of the twelve subtasks, successes/failures,
  contribution boundaries, and next steps.
- `historical_summary.csv`: historical formal tables, development pilots, and ablation
  summaries, with original paths and SHAs; explicitly marked `raw_evidence_included=false`.
- `artifact_dependencies.json`: inventory of existing configuration references to `results/`,
  `.runtime/`, and similar paths, separating outputs from inputs that must be rebuilt.
  Historical configurations are included; this is not a queue requiring every job to rerun.
- `rebuild_plan.json`: reconstruction order, directly runnable public source entry points,
  and stages that still require method repairs.
- `source_manifest.json`: current main/nested repository versions and uncommitted state.
- `environments/`: actual package versions of six source-device environments; restore
  `baseline_tasks_torch` first.
- `media/`: GIFs, preview images, and provenance for the first environment's three subtasks.
  These display historical trajectories; they are not raw episodes or fresh validation.
- `cold_start_validation.json`: isolated validation results from a mirror without results.
- `rsync.exclude`: transfer rules excluding top-level `results/`, `.runtime/`, and caches.
- `SHA256SUMS`: checksums for this handoff directory; the complete source archive has its
  own archive SHA.

See the [twelve-subtask roadmap](../icdt-paper-story-and-experiment-roadmap.md) for the
research narrative and [architecture boundaries](../cross-ebdm-architecture-refactor.md)
for architectural rules. The [full migration plan](../icdt-device-migration-handoff.md)
remains available when retaining original artifacts. The cold-start order here takes
precedence over its instructions to restore and run old actors.

## Transfer

The prepared `icdt-cold-start.tar.gz` is a source/asset working-tree snapshot excluding
top-level `results/` and `.runtime/`. It preserves uncommitted and untracked files and
nested repositories, including this directory and the GIFs. Transfer the archive and
its `.sha256` file to the new device. Verify and extract:

```bash
sha256sum --check icdt-cold-start.tar.gz.sha256
mkdir -p /YOUR/WORKSPACE
tar -xzf icdt-cold-start.tar.gz -C /YOUR/WORKSPACE
cd /YOUR/WORKSPACE/genedynamics
sha256sum --check --quiet docs/development/icdt-cold-start/SHA256SUMS
```

Alternatively, transfer the current working tree directly from the old device:

```bash
cd /home/hectorh/workspace/genedynamics
rsync -a --no-owner --no-group --partial --info=progress2 --mkpath \
  --exclude-from=docs/development/icdt-cold-start/rsync.exclude \
  ./ NEW_USER@NEW_HOST:/YOUR/WORKSPACE/genedynamics/
```

Transfer into a new, empty directory. Do not use `--delete` to clean an old workspace,
and do not rely solely on `git clone`. Derived ACP/FADA assets from the old `.runtime/`
are excluded from this package. Neither method currently carries a main-table role;
if their native routes are restored later, download or rebuild assets according to
their source/resource locks.

## Restore the Environment

The source platform is Linux x86_64, Python 3.12.3. The main task environment uses
Torch 2.5.1+cu124 and MuJoCo 3.6.0. The destination GPU is unknown. These instructions
restore the source environment's pinned versions; incompatible new hardware requires
an explicit environment revision and renewed qualification. Native CPU physics does
not require JAX to use the GPU.

```bash
python3.12 -m venv .runtime/baseline_tasks_torch/venv
export ICDT_PY="$PWD/.runtime/baseline_tasks_torch/venv/bin/python"
"$ICDT_PY" -m pip install 'setuptools>=77' wheel
"$ICDT_PY" -m pip install 'torch==2.5.1+cu124' \
  --index-url https://download.pytorch.org/whl/cu124
"$ICDT_PY" -m pip install \
  -r docs/development/icdt-cold-start/environments/baseline_tasks_torch.requirements.txt
"$ICDT_PY" -m pip install -e . --no-deps
"$ICDT_PY" -m pip check
"$ICDT_PY" -m genedynamics.solvers.single.cross_ebdm.learning.backends.native.build \
  --output .runtime/cross_ebdm/native/libcross_ebdm.so
export CROSS_EBDM_NATIVE_LIBRARY="$PWD/.runtime/cross_ebdm/native/libcross_ebdm.so"
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cpu
```

A system C++17 compiler is required. Record the new library SHA, GPU UUID, and source
identity in the new execution protocol. Old configuration GPU UUIDs and library SHAs
do not apply to the new device; do not alter old reports to pass validation.
Do not substitute the generic D3IL setup for this pinned restoration, and do not run
old formal evaluation entry points directly.

## Cold-Start Execution Order

1. **Environment and physics.** Check dependencies, robot assets, the contract loader,
   and evaluator. Public source controllers and physical scenes should construct
   without `results/`. Start on CPU; use GPU0 for subsequent learning.
2. **Repeat public source qualification.** Aperture, sequential, and corner have
   qualification entry points independent of learned checkpoints. The commands below
   create new outputs and development seeds without overwriting old protocols/results.
3. **Repair and rebuild methods.** Prioritize real task-band, response-term, and selective-
   gate implementation for corner. Rebuild all training data, response models, ICDT,
   three ablations, and baseline actors. Give ASID/AdaptSim corresponding task training.
   Files referenced under `results/` by old actor configurations will not restore themselves.
4. **New paired development comparisons.** Prioritize corner's sustained-turn/high-friction
   and slow-contact-response cells. Use fresh paired resets, independent training seeds,
   and a common budget; retain failures. Implement edge pivot on CPU in parallel.
5. **New formal freeze.** Freeze a new cohort after method and source/nominal qualification.
   Register training instances, reset seeds, previously accessed test requests, and artifact
   identities explicitly. Do not append new runs to old logs.

Public source qualification examples (run at the repository root on the new device):

```bash
"$ICDT_PY" - <<'PY'
import json
from pathlib import Path
import subprocess, sys, tempfile

recipes = [
    ('configs/arm/obstacle_transfer/validation/aperture_brake_cpu.json',
     'scripts/paper/cross_ebdm/qualify_aperture_brake.py'),
    ('configs/arm/obstacle_transfer/validation/sequential_aperture_reorientation_cpu.json',
     'scripts/paper/cross_ebdm/qualify_sequential_aperture_reorientation.py'),
    ('configs/arm/guided_contact_alignment/validation/corner_guided_seating_cpu.json',
     'scripts/paper/cross_ebdm/qualify_corner_guide.py'),
]
with tempfile.TemporaryDirectory(prefix='icdt-source-plans-') as temporary:
    for index, (source, entry) in enumerate(recipes):
        plan = json.loads(Path(source).read_text())
        plan['output_dir'] += '_cold_start'
        for offset, episode in enumerate(plan['episodes']):
            episode['seed'] = 12000000 + index * 100 + offset
        destination = Path(temporary) / Path(source).name
        destination.write_text(json.dumps(plan, indent=2))
        subprocess.run([sys.executable, entry, '--config', str(destination)], check=True)
        report = json.loads((Path(plan['output_dir']) / 'report.json').read_text())
        if not report.get('passed', False):
            raise SystemExit(f"Source qualification failed: {plan['task']}")
PY
```

These checks establish public source feasibility, not cross-embodiment contributions:
the original aperture recipe uses Panda; sequential and corner use xArm7. Actual
source-to-target evidence requires separate experimental design. New-seed recipes are
not replicas of the old 4/4 source outcomes; record their actual outcomes. If a previous
attempt already created `_cold_start` outputs, review its success/failure records before
creating an explicit revision. Do not delete failures and rerun.

There is currently no trustworthy single command to retrain every baseline and produce
a formal twelve-task table from scratch. Existing training scripts still depend on old
source banks, datasets, and checkpoints. Rebuild these incrementally using
`artifact_dependencies.json` and `rebuild_plan.json`. Do not create placeholder artifacts
to bypass validation.

## Established Findings and Pitfalls to Avoid Repeating

| Family | Preserved research findings | Priority |
|---|---|---|
| Obstacle | All three subtasks implemented; historical base-task success nearly saturated; two-request pilots for each new subtask; new 100-episode formal evaluation unexecuted | Rebuild, fix task versions, and complete formal comparison with matched methods |
| Guided contact | Corner public source 4/4; old phase multipliers are not valid mechanism ablations; edge pivot unimplemented | Real training/three ablations, a physical pivot, and angular response |
| Connector assembly | Jam: ICDT 42.19%, AdaptSim 62.81%; all ICDT failures are overforce; compliant's old qualification SHA mismatches | Shared unloading and delayed force consequences; requalify fixture, then add twist locking |
| Support-coupled | Stance ICDT 86.88% shows system-level performance; reposition pilot runs; disturbance recovery unimplemented | Morphology demonstration, independent instances, physical disturbance recovery |

Aperture full and all three matched ablations select multiplier 1.25: all achieve 4/4
and normalized time 0.92. The current response objective participates in candidate
scoring, but the runtime actor remains a phase-wise translation multiplier. It does
not implement the manuscript's complete history/model-support gating. The main pilot's
old task-only actor is also different, so 2/2 versus 0/2 does not directly establish
transport gains. Sequential's final residual acts only in the terminal phase;
crossing/adjacent-alignment failures remain method limitations.

Next, revise the theory around event-sufficient responses, actual task bands/gates,
phase-error composition, execution-layer attribution, and complete target-interaction
budgets. Local wins and time-force tradeoffs are acceptable; universal dominance is
not required. Formal Franka-to-UR results remain unfinished. All three GIFs in this
directory show existing xArm7 trajectories.

The GIFs replay recorded qpos/qvel without advancing physics. Both aperture tasks use
explicitly labeled translucent visualization; physical geometry is not removed.
Numerical evaluations and accounting for all three trajectories match when recomputed
with the current evaluator; saved-pose FK errors in the display models are all 0.
Dual-route and aperture have historical evaluator source SHAs that differ from the
current version; their complete audit `passed=false` values remain unchanged in the
media provenance JSON. Sequential also matches the current source identity. Replay
checks must not be described as newly completed formal evaluations.

A temporary mirror without `results/` and `.runtime/` completed 10 CPU control steps
each for aperture, sequential, and corner. It reused Python third-party dependencies
installed on the source device. This does not establish successful installation on
a fresh machine or completion of full source episodes using the three new seeds.
See `cold_start_validation.json`.

Planned main table: ICDT, task-only, ASID, AdaptSim, response MPC, DIAL-MPC.
Four main metrics: constrained success, critical-event completion, peak force, and
failure-penalized normalized time. Three ablations: no-calibration,
no-response-anchoring, unconditional-response-anchoring. ACP/Human2Sim2Robot remain
outside the table; poor scores from unqualified FADA adaptations are not failures
of the original paper's method.

## Instructions for a New Session

> This is the ICDT / RSS 2027 cold-start working tree; old results and runtime were not
> transferred. Read this README, research_state.json, artifact_dependencies.json, and
> rebuild_plan.json first. Historical summaries preserve research memory, not locally
> re-auditable raw evidence. Do not run formal configurations referencing missing old
> checkpoints or interpret planned/released states as completed. Rebuild the environment
> and public source qualification, then advance genuinely matched ICDT/three-ablation
> training and fresh paired development for corner. Implement edge pivot on CPU in
> parallel. Maintain GPU0-only. Follow the original architecture: algorithms in their
> respective solvers, physics/evaluation in tasks/envs, high-level learning free of
> algorithm clutter, thin paper orchestration scripts, and configs/results within the
> existing task directories. Name new files semantically. Keep the manuscript an academic
> draft and development records in docs. Do not commit without instruction.
