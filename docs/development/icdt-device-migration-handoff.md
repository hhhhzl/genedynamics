# ICDT Device Migration and Research Handoff

Verified: 2026-11-18, America/New_York. Target venue: RSS 2027.

The user's subsequent lightweight option is documented in the
[cold-start handoff without transferring results](icdt-cold-start/README.md).
It includes research state independent of `results/`, reconstruction dependencies,
environment inventories, and GIFs of three tasks. The following is the full migration
option retaining all old artifacts. Snapshots mentioned here record the versions at
creation time; use the newer cold-start package checksums for subsequent renderer
and cold-start material updates.

This document is the starting point for a new device/session. It records migration
procedures, current evidence levels, and the next development revision. This handoff
work only organizes documentation: no training, criterion changes, Git commits, or
changes to old results. The research design follows the
[twelve-subtask roadmap](icdt-paper-story-and-experiment-roadmap.md). Where that document's
earlier states mention “eight tasks,” “all four new tasks unimplemented,” or a “complete
closed loop,” the verification here takes precedence.

## 1. What Must Be Transferred

Transfer the complete working tree, preserve relative paths, and use the supplied
exclusion list to omit reconstructible runtime environments. Pushing/cloning Git alone
omits experiment results, manuscripts, baseline source, and the latest untracked work.

| Content | Approximate local size (du) | Treatment |
|---|---:|---|
| `results/` | 117 GiB, approximately 233,000 files | Retain everything, including failures, invalid labels, raw evidence, checkpoints, statistics, and source snapshots |
| `genedynamics/`, `configs/`, `scripts/`, `test/`, `docs/`, `requirements/`, and root files | Relatively small | Preserve modified/untracked files and `.git/` |
| `latex/` | 303 MiB | Retain in full; `latex_icdt` and `latex_mga` each contain a nested Git repository |
| `baselines/` | 754 MiB | Preserve author source, revisions, and local changes; the main repository ignores this directory |
| `third_party/` | 2.1 GiB | Preserve assets; Menagerie's `unitree_go2/go2_mjx.xml` has local modifications |
| `.runtime/` | 57 GiB, including environments and experiment assets | Exclude venvs and pip/uv/compiler caches; retain native, assets, derived_oracle, locks, and installation records |

`results/arm/` is approximately 39 GiB; `results/quadruped/`, 7.7 GiB;
`results/_development/`, 55 GiB; and `results/_qualification/`, 16 GiB.
Historical directories may still contain the provenance of current checkpoints;
do not delete them simply because they are old. Reserve at least 180–200 GiB on
the new device, and more for multiple native baseline environments and new experiments.
These are storage-planning estimates, not compressed archive sizes.

The rsync dry-run with the exclusion list totals approximately **128.8 GiB (138.3 GB)**,
including approximately 261,000 regular files. Retained `.runtime/` experiment dependencies
occupy about 8.0 GiB. All 1,221 retained symlinks currently resolve, with no final targets
outside the project or in excluded runtime environments. Absolute links still require
handling if the root path changes.

In particular, `.runtime/acp_native/assets/` (approximately 1.6 GiB) and
`.runtime/fada_native/derived_oracle/` (approximately 1.9 GiB) must not be excluded
along with virtual environments. Recreate old virtual environments after transfer;
they contain absolute paths and machine-specific binaries.

## 2. Generated Handoff Snapshot

Location: `results/_runtime/device_migration_20261118/`.

- `snapshot.json`: main/nested repository versions and state, directory inventory,
  symlinks, and report states.
- `git/`: working-tree status and binary diffs of tracked changes for review;
  full rsync preserves untracked files.
- `environments/`: Python versions and package inventories for available environments;
  credentials and complete environment-variable dumps are excluded.
- `rsync.exclude`: excludes only explicit runtimes/caches and local credential directories;
  it does not filter through Git ignore rules.
- `SHA256SUMS`: SHA-256 hashes of current source, configurations, development documents,
  manuscript source, and selected critical artifacts. This is a critical-dependency
  checksum list, **not a complete SHA inventory of all 117 GiB of results**.

All 3,113 listed entries were verified on the source device. Transfer exclusions and
Bash syntax in the instructions were checked. Transfer, installation, testing, and
physics smoke checks on the destination device have not been executed.

The main repository baseline commit is `9deb25e85896418655b1f26ae3eb7dd4cba8b6cc`,
with uncommitted working-tree changes. The `latex/latex_icdt` baseline is
`c82b87bcae99f09910378d411fc3b2af6fc848b9`, with six modified sections. Copying the
complete working tree preserves these changes; migration does not require a commit.

No experiment process matching this project was found at inspection time, and the
`nvidia-smi` compute-process query was empty. This describes only that inspection.
Stop new experiment writes during the actual transfer and perform a final incremental
sync before acceptance.

## 3. Transfer and Integrity Checks

Run these commands on the old machine. They assume a Linux destination with rsync,
SSH access, and a writable destination directory. Replace `NEW_USER@NEW_HOST` with
the actual login/address. Prefer the original absolute path. If using a different
path, change the destination below, then handle path bindings as described in Section 4.

```bash
cd /home/hectorh/workspace/genedynamics
export ICDT_DEST_HOST='NEW_USER@NEW_HOST'
export ICDT_DEST_ROOT='/home/hectorh/workspace/genedynamics'

# --mkpath requires receiver support; on older rsync, create ICDT_DEST_ROOT there first.
rsync -a --no-owner --no-group --partial --info=progress2 --mkpath \
  --exclude-from=results/_runtime/device_migration_20261118/rsync.exclude \
  ./ "${ICDT_DEST_HOST}:${ICDT_DEST_ROOT}/"

# After a disconnect, rerun the same command; do not add --delete.
# Once writes stop, do the final sync above, then compare file contents at both ends:
rsync -a --no-owner --no-group --checksum --dry-run --itemize-changes \
  --exclude-from=results/_runtime/device_migration_20261118/rsync.exclude \
  ./ "${ICDT_DEST_HOST}:${ICDT_DEST_ROOT}/"
```

The checksum dry-run reads every included regular file on both devices; duration
depends on storage speed. If contents differ, first confirm the source is stable,
then remove `--dry-run` for checksum synchronization and verify again. Resolve content,
missing-file, and symlink differences. Ownership is explicitly ignored; explain any
other metadata differences. This check does not delete extra destination files, so
use an empty directory or one dedicated to this migration. Avoid concurrent commit,
fetch, gc, or other Git writes while copying `.git/`.

On the new machine, before modifying code or rebuilding native libraries:

```bash
cd /home/hectorh/workspace/genedynamics
sha256sum --check --quiet results/_runtime/device_migration_20261118/SHA256SUMS
git status --short
git -C latex/latex_icdt status --short
git -C third_party/mujoco_menagerie status --short
```

If a removable drive is the only option, replace the destination with a directory
on that drive in the same rsync command. Prefer a Linux filesystem to preserve
symlinks. Use a tar container on media lacking symlink support; do not dereference
links into files with unclear provenance. Retain originals on the old device until
migration acceptance is complete.

## 4. Restore Paths, GPU Bindings, and Environments

The source platform is Ubuntu 24.04.3, x86_64, Python 3.12.3, with an RTX 4500 Ada GPU
and driver 570.153.02. The current Torch task environment is
`.runtime/baseline_tasks_torch/venv`, using Torch 2.5.1+cu124 and MuJoCo 3.6.0.
The destination GPU and driver are unknown. The following restores the existing
research environment and has not been validated on the destination.

Restore this task environment first; create other native baseline environments as
needed. `scripts/setup/setup.sh` defaults to the full D3IL environment and installs
different Torch/JAX versions, so it is unsuitable for exact restoration here.
Old `.runtime/*/requirements.lock.txt` files may include editable Git URLs for old
commits. Do not execute those URLs unchanged; install the transferred working tree
with `pip install -e . --no-deps`.

```bash
cd /home/hectorh/workspace/genedynamics
python3.12 -m venv .runtime/baseline_tasks_torch/venv
export ICDT_PY="$PWD/.runtime/baseline_tasks_torch/venv/bin/python"

"$ICDT_PY" -m pip install 'setuptools>=77' wheel
"$ICDT_PY" -m pip install 'torch==2.5.1+cu124' \
  --index-url https://download.pytorch.org/whl/cu124
"$ICDT_PY" -m pip install \
  -r results/_runtime/device_migration_20261118/environments/baseline_tasks_torch.requirements.txt
"$ICDT_PY" -m pip install -e . --no-deps
"$ICDT_PY" -m pip check
```

The observed requirements inventory excludes the project itself; Torch is installed
first from its wheel index. If a pinned version is unavailable or old Torch does not
support the new GPU, record the precise conflict and create a new environment revision.
Do not silently upgrade and call it the original environment. The repository's
`requirements/baselines/tasks-pytorch.txt` specifies direct dependencies, not a complete
lock of the original environment's transitive dependencies. Snapshot inventories allow
more precise restoration. An offline destination needs a wheelhouse prepared separately
on the source device; one has not yet been downloaded.

Several frozen protocols bind the SHA of
`.runtime/cross_ebdm/native/libcross_ebdm.so`; transfer it. If compilation is needed
on the new device, use a new filename and retain the original binary and SHA:

```bash
"$ICDT_PY" -m genedynamics.solvers.single.cross_ebdm.learning.backends.native.build \
  --output .runtime/cross_ebdm/native/libcross_ebdm_migrated.so
export CROSS_EBDM_NATIVE_LIBRARY="$PWD/.runtime/cross_ebdm/native/libcross_ebdm_migrated.so"
```

Building requires a C++17 compiler. Some runners hard-code old paths/SHAs and may
ignore the environment variable. Check each entry point before execution and bind
the new library explicitly in new run contracts. Do not overwrite hashes in old reports.

Likewise, configurations contain old-machine GPU UUIDs;
`CUDA_VISIBLE_DEVICES=0` cannot repair those bindings. Maintain GPU0-only on the new
device, record its GPU0 UUID, and create migrated development/execution configurations,
retaining originals and original reports. Do not spoof old UUIDs or bulk-replace
paths/hashes in old artifacts to pass checks.

The inventory found absolute symlinks to the old root in historical pytest snapshots.
If the root changes, inspect snapshot `symlinks` individually and use external path
mapping or a compatible root. Keep original evidence, checkpoints, and source snapshots
unchanged. Retain Git metadata for third-party source and runners; merely recloning
the upstream default branch is insufficient.

## 5. Minimum Post-Migration Acceptance

Start with integrity checks and CPU offline re-evaluation, then check GPU availability
and perform an independent smoke test. The offline re-evaluation below only reads
existing evidence; it launches no physics episodes and opens no formal test requests:

```bash
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cpu
"$ICDT_PY" -m pytest -q \
  test/transfer/test_subtask_contracts.py \
  test/transfer/test_aperture.py \
  test/transfer/test_semantic_method_binding.py \
  test/learning/test_semantic_phase.py
"$ICDT_PY" scripts/paper/cross_ebdm/verify_results.py --evidence \
  results/arm/obstacle_transfer/validation/aperture_brake_response_paired_pilot/evidence/cross_ebdm/seed_948000.json.gz \
  results/arm/obstacle_transfer/validation/sequential_aperture_response_paired_pilot/evidence/cross_ebdm/seed_948100.json.gz
"$ICDT_PY" -c 'import torch, mujoco; print(torch.__version__, mujoco.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Then create new migration-smoke outputs under an existing task's `validation/` and
run source/nominal checks. Old source-qualification defaults already contain results;
do not rerun and overwrite them. Different hardware may produce numerical differences;
record destination-side migration qualification separately. Passing tests establishes
runnable code/interfaces, not validated paper mechanism contributions.

## 6. Current Findings, Failures, and Evidence Boundaries

The historically valid formal batch is GPU0 `independent_main_table_formal_v2`:
36 method-by-task cells, 232 actor jobs, and 11,600 outcomes. It remains valid under
its original frozen protocol. Its table is not the new equal-budget twelve-subtask
comparison. Target fixed arms are xArm7; the support-coupled body is Go2+Z1.

| Historical formal success (%) | Dual-route | Dual-channel | Jam recovery | Stance insertion |
|---|---:|---:|---:|---:|
| ICDT | 100.00 | 98.75 | 42.19 | 86.88 |
| Task-only | 100.00 | 97.50 | 47.19 | 60.16 |
| AdaptSim | 98.44 | 99.06 | 62.81 | 86.72 |
| ASID | 96.88 | 98.44 | 59.69 | 87.50 |
| Response MPC | 100.00 | 99.69 | 48.44 | 70.78 |
| DIAL-MPC | 100.00 | 0.00 | 36.25 | 83.75 |

Achievements: historical comparisons cover all four families; new physical subtasks
have runnable implementations, paired resets, and evidence audits. Limitations: simple
tasks nearly saturate; ICDT's 185/320 jam failures are all overforce; stance gains over
task-only are confounded by execution-layer differences and cannot be attributed solely
to transport. Adaptation failures such as DIAL's must not be described as definitive
mechanism failures of the original papers on these tasks. These values come from the
retained historical protocol; relabeling columns does not make them a new main table.

| Family | Subtask state | Next step |
|---|---|---|
| `obstacle_transfer` | `dual_route_carry` has historical formal results; `aperture_brake` and `sequential_aperture_reorientation` each have source 4/4 with matching current contract SHAs and completed two-request pilots per method | Provisionally fix task versions; correct method-comparison identities before new formal evaluation |
| `guided_contact_alignment` | `dual_channel_contact` has historical formal results; `corner_guided_seating` source 4/4 with matching contract; `edge_slide_pivot` is design-only | Prioritize genuine matched corner training/ablations; implement edge pivot on CPU in parallel |
| `keyed_insertion_recovery` and `compliant_keyed_insertion_recovery` | `jam_recovery` has historical formal results; `compliant_fixture_insertion` has old source 4/4 but a mismatched current contract SHA; `insert_then_lock` is design-only | Requalify fixture; repair shared unloading and delayed force consequences; build a physical bayonet and retention test for twist locking |
| `mobile_keyed_insertion_recovery` | `stance_insertion` has historical formal results; `reposition_then_dock` source 4/4 with matching contract and all five methods succeeding on two requests; `disturbed_docking_recovery` is design-only | Add independent instances and physical body-disturbance recovery for morphology demonstration |

Nine of twelve subtasks are implemented; three remain designs. Implementation does
not imply formal multi-instance results.

Latest obstacle evidence: aperture ICDT and AdaptSim both achieve 2/2, with normalized
times 0.92 and 0.985; task-only, ASID, and response MPC achieve 0/2. This task-only
is an old dense actor, not a matched ablation of the new ICDT. The genuinely matched
full/no-calibration/no-response/unconditional treatments all choose multiplier 1.25,
achieve 4/4, and take time 0.92. No independent response-anchoring benefit has been
identified. All five sequential methods achieve 2/2; ICDT time is 0.8709 versus AdaptSim
0.8073. Final transport acts only in the terminal phase after clearing both frames.
Retain development crossing/alignment failures.

The obstacle release report is `released_for_formal_execution`. The new 100-episode
batch has opened 0 formal resets, and its formal output directory does not exist.
This is an executable protocol awaiting execution, not completed formal results.
Old corner/fixture phase-multiplier experiments remain development diagnostics;
they cannot be promoted to genuine ICDT ablation evidence.

## 7. Evidence Entry Points to Retain

All paths are relative to the repository root. Full `results/` retains the underlying
evidence referenced by these entries.

- Valid formal table: `results/_analysis/four_task_transfer/independent_main_table_formal_v2/`
  (`main_table.json`, `uncertainty.json`, `protocol.json`).
- Execution state: `results/_analysis/four_task_transfer/independent_main_table_execution_v2/state.json`.
- Old invalid batch: `results/_analysis/four_task_transfer/independent_main_table_execution_v1/invalidation.json`
  and `results/_protocol/four_task_transfer/invalid_runs/`. Retain invalidation reasons;
  do not merge these outcomes into the valid table.
- Overforce diagnosis: `results/_analysis/four_task_transfer/keyed_insertion_overforce_diagnosis_20261105/`.
- New aperture training: `results/arm/obstacle_transfer/training/aperture_brake_response_transport/`.
- New paired pilots: `results/arm/obstacle_transfer/validation/aperture_brake_response_paired_pilot/`
  and sibling `sequential_aperture_response_paired_pilot/`.
- Pending execution release: `results/arm/obstacle_transfer/validation/three_subtask_formal_release_v1/report.json`.
- Source qualification: each family's `validation/<semantic_task>/report.json`.
- Corner development diagnostics: `results/arm/guided_contact_alignment/validation/corner_guided_seating_paired_development/`.
- Method implementation boundary: `genedynamics/solvers/single/cross_ebdm/learning/semantic_phase.py`;
  training entry: `scripts/paper/cross_ebdm/train_aperture_brake_response_transport.py`.

The main manuscript is `latex/latex_icdt/`, and prior-work reference is `latex/latex_mga/`;
transfer both. `/home/hectorh/pascal/brainstorm.md` was not found during this check,
so the original external file cannot be claimed as packaged. Implemented manuscript
ideas are in the twelve-subtask roadmap. Preserve the original brainstorm separately
if it still exists elsewhere on the old device.

## 8. Next Development Order and Theory Revisions

1. **Migration acceptance.** Compare working trees, critical SHAs, and all transferred
   contents; recreate the task environment, bind the new GPU0, and perform offline
   re-evaluation and independent smoke tests. Do not use old formal requests for debugging.
2. **Corner method alignment.** Full and three ablations must share model capacity, data,
   optimization budget, and execution permissions, with actual task-band/gate objective
   differences. Current semantic v2 selects translation multipliers from finite candidates
   using a linear one-step velocity model. It does not implement the complete history/
   model-support gating described in the manuscript. Align code and theory with actual
   implementation; metadata changes alone are insufficient.
3. **Critical-contact experiments.** Reuse corner's sustained-turn/high-friction and
   slow-response cells first. Compare ICDT, three ablations, task-only, ASID, AdaptSim,
   and response MPC on fresh paired resets and independent training seeds. Add DIAL-MPC
   after relevant interface and source/nominal qualification. Assess reproducible
   success/event/force/time tradeoffs without requiring wins everywhere.
4. **Complete tasks in parallel.** On CPU, implement edge slide pivot's asymmetric object,
   physical pivot, and anti-bypass evaluator. Recheck the compliant contract; prioritize
   shared unloading and delayed force consequences for jam. Then implement insert/lock,
   and finally body-disturbance recovery; do not expand into locomotion/navigation.
5. **Finish obstacle formal evaluation.** Execute the two specified 50-episode plans after
   verifying method identities, resources, and new-machine execution bindings. Explicitly
   revise protocols if methods/tasks change; do not present revised outcomes as the old
   frozen version. The existing 100-episode design is not a complete formal table over
   independent training instances.
6. **Paper-level comparisons.** Complete the six-row table over twelve subtasks: ICDT,
   task-only, ASID, AdaptSim, response MPC, DIAL-MPC. Unqualified methods remain explicitly
   pending, not zero-scored. ACP/Human2Sim2Robot remain outside the table. Use FADA only in
   qualified targeted comparisons. Limit main ablations to no-calibration,
   no-response-anchoring, and unconditional-response-anchoring.

Theory priorities: response features covering event timing/order, contact, angular
response, and delayed consequences; notation for task bands, history compatibility,
model support, and phase gates that maps to executable rules; conditional bounds with
explicit source margin, calibration error, anchoring error, and unsupported mass.
Phase-error composition and execution-layer effects on attribution still need work.
Budget curves must count probes, model selection, complete task rollouts, and failures,
with independent episode/instance splits. Improvements from having more candidates
must not all be attributed to model calibration.

Current evidence does not support “transport consistently outperforms strong baselines,”
and formal Franka-to-UR evidence remains unfinished. Preserve matched ablations with
no gain and narrow the associated claims. Do not manipulate baselines to meet a venue goal.

## 9. Ready-to-Use New-Session Instructions

> This is the GeneDynamics / ICDT (RSS 2027) working tree migrated from the old device.
> First read `docs/development/icdt-device-migration-handoff.md`,
> `docs/development/icdt-paper-story-and-experiment-roadmap.md`, and
> `docs/development/cross-ebdm-architecture-refactor.md`; check
> `results/_runtime/device_migration_20261118/` against the actual artifacts.
> Nine subtasks are implemented and three remain unimplemented. Retain the valid
> historical v2 main table; the new 100 obstacle episodes have not run. Complete
> migration acceptance first, then advance genuine matched ICDT/three-ablation
> training for corner and CPU physics for edge pivot. Maintain GPU0-only with
> new-machine resource bindings and do not touch other users' jobs. Follow the
> original architecture: algorithms in their own solvers, physics/evaluation in
> tasks/envs, high-level shared learning, thin paper scripts, and configs/results
> in existing task directories. Use semantic filenames and preserve old SHA-bound
> artifacts. Keep academic content in the manuscript and progress/failures in
> development documents. Do not commit unless requested.

Conversation history does not transfer automatically. This document and the on-disk
artifacts jointly carry the context.
