# V1 validation evidence

This page records the release-candidate checks from 2026-10-05 and the
documentation/package refresh on 2026-10-09, on branch
`release/v1-open-source`. It is validation evidence, not a performance claim.

## Release surface

- 80 experiment configurations and 16 deployment configurations validated.
- 16 method registrations and 7 environment configuration keys across six
  task families matched the machine-readable V1 catalogue.
- Soft-robot, co-design, 3DGS, MBD3D, MRMFMBD, and morphology paths were absent
  from the source and wheel inventories.
- Humanoid push-to-line and MGA GPU remain outside the readiness claim.

## Test evidence

The earlier development-checkout CPU Docker regression produced:

```text
725 passed, 19 skipped, 152 deselected, 0 failed
```

The deselected cases carry explicit slow, Docker, accelerator, Brax, or Isaac
markers. After making D3IL registration lazy, the affected regression added:

```text
6 passed, 2 optional-dependency skips, 0 failed
```

The validated numerical stack was Python 3.10.20, JAX/JAXlib 0.6.2, NumPy
2.2.6, and SciPy 1.15.3. CI repeats the portable core subset on Python 3.10
and 3.12.

## Package evidence

The latest local validation wheel, built from a clean export of `ea04c66`, has:

| Property | Value |
| --- | --- |
| Filename | `genedynamics-0.1.0-py3-none-any.whl` |
| Size | 12,195,764 bytes |
| Files | 809 |
| SHA-256 | `7f83ddc9cfd53aa8741272072d278b89e42d029415cb6ad2dcd14057d46219ca` |
| Required package assets missing | 0 |
| Forbidden V1 paths | 0 |
| Twine metadata check | Passed |

The wheel and all core dependencies were installed into a new Python 3.12
virtual environment on macOS arm64. `pip check`, package/version import, JAX CPU
backend discovery, optional D3IL plugin import, and the documented planar
dry-run all passed from outside the repository checkout.

The wheel and source archive include all 13 declared license files, verified
against the checked-in full texts. All 683 packaged Python files match the
preceding CPU-tested wheel. Robot XML changes contain attribution comments
only; parsed XML content is unchanged. Published CI artifacts receive their
own `SHA256SUMS`; the local checksum above is not a claim about a future rebuild.

The source and wheel inventory is stored in
`release/v1_inventory.json`. The core dependency license closure is stored in
`release/v1_dependency_licenses.json`.

## Static and documentation evidence

- MkDocs built in strict mode.
- 1,018 release Python files parsed with zero syntax errors.
- `git diff --check` reported no whitespace errors.
- Credential pattern scanning reported zero AWS, GitHub, OpenAI, Slack, or
  private-key signatures.
- The current source inventory reports three files above 10 MiB: two D3IL
  camera meshes and the homepage GIF. None is included in the wheel.
- The 20-package core dependency closure reported no detected strong-copyleft
  license.

## Documentation refresh — 2026-10-09

- The homepage now uses 44 original clips in five groups: MDOC, MD-COAS, 2GO,
  MGA and hardware. Titles, subtitles and rulers were cropped from the inputs;
  the camera composition adds no text. All five groups and camera views were
  visually reviewed.
- The 24-second GIF is 896 × 661 at 6 fps (21.57 MiB). The same composition is
  available as a 1344 × 992, 12 fps MP4 (6.59 MiB). This is presentation playback,
  not a runtime or latency measurement.
- All 44 input sizes and SHA-256 checksums matched
  `docs/assets/showcase_sources/v2/manifest.json`. The exports regenerate from
  these checked-in inputs without the original research folders or simulators.
- MD-COAS pairs three diffusion views with native animations of 20 candidate
  trajectories: planar L6/seed 0, planar L10/seed 8 and arm avoidance L1/seed 0.
  The arm candidates are TCP projections of planned 7-DoF trajectories. Source
  GIF hashes were verified; every clip occupies exactly one gallery tile.
- The new Python example was run on CPU from the rebuilt wheel, outside the
  repository checkout: **48 steps, goal error 0.099**, with finite-value and
  goal-arrival assertions passing. `pip check` and Twine metadata checks passed.
- Planner descriptions were checked against original papers. CFS-MBD is
  documented as MD-COAS's historical implementation name.
- Environment and constraint catalogues document six task families, seven
  configuration keys and four numerical constraint-solver interfaces.
- The framework positioning and architecture now connect policy learning,
  learned trajectory diffusion, model-based generative inference and control.
  The learning guide documents training, checkpoint reuse and offline reliability
  calibration without claiming automatic online retraining.
- The roadmap is a concise checklist, and the architecture uses a coordinated
  learning/task/execution palette. The gallery preserves native footage and a
  text-free composition.
- The documentation built in strict mode. README links, anchors and Python
  syntax passed static checks. Training-guide commands were checked against
  their parsers; no new training runs were performed. Runtime source, task
  configurations and tests are unchanged by this refresh; the earlier
  regression remains the runtime evidence.


## Manifold positioning and batched constraints — 2026-10-09

- The homepage follows “Generative dynamics on manifold” and shows the complete
  task-to-execution workflow: task definitions, planners, learning, constraints,
  the planning workflow, simulation, deployment and evaluation. A rendered PNG
  accompanies the editable SVG, with Learning explicitly connected to planner
  priors and offline evaluation feedback.
- The obsolete 2GO corridor and stepping-stone poster images were removed; the
  unified demonstration gallery is unchanged.
- The batch guide documents candidate, horizon and refinement axes, CBF/CFS and
  manifold feedback, MGA learned proposals, and DPCC/SafeDiffuser integration
  boundaries. Its JAX example ran with shape `(64, 20, 2)` and maximum residual
  `2.2351741790771484e-08`.
- README checks passed for 70 local links, 14 stable section anchors and Python
  syntax. MkDocs built in strict mode. The rebuilt wheel passed Twine and differs
  only in metadata from the preceding wheel; runtime files are unchanged.

## CI checkout repair — 2026-10-09

- The reported GitHub failure occurred before testing: recursive checkout could
  not fetch Menagerie commit `e5146679f3cfcb327cf759fc9706f5bb0236bd5e` from
  its official remote. This was a local asset adaptation commit.
- The gitlink now pins its official parent
  `a03e87bf13502b0b48ebbf2808928fd96ebf9cf3`, verified through a shallow fetch
  from the official repository. The Go2 MJX changes are preserved as an explicit
  patch. Setup was checked for first application, repeat application and
  conflict refusal; the patched XML matches the original local version exactly.
  The existing local submodule checkout was left untouched.
- A new Python 3.12 core environment ran the workflow's selected tests with
  coverage: **89 passed, 4 skipped**. The selected simulation tests ran in a
  temporary macOS environment: **20 passed, 1 skipped**, including the G1
  TorchScript control path. The D3IL startup test skipped for optional dependencies.
- The Ubuntu simulation workflow now installs CPU Torch and `urchin` explicitly
  and applies the asset patch after checkout. Test selections remain enabled.
  These local checks do not claim that the new remote Actions run has completed.


## Licensing and publication preparation — 2026-10-09

- The project owner selected MIT for original framework code.
- Bundled D3IL, Diffuser/DPCC and SafeDiffuser notices were restored. Robot
  models and the third-party G1 policy retain Apache-2.0 or BSD-3-Clause terms.
- License texts are included in distribution metadata and package assets. The
  wheel license expression reflects its MIT, Apache-2.0 and BSD-3-Clause
  components; separately installed optional dependencies retain their own terms.
- Excluded research source is absent from V1. An additional 294 untracked
  research caches and residual files were removed. The owner requested preserving existing Git history; publication uses the
  same repository and excludes these systems from the V1 branch and packages.

## Open gates

- Publication-grade latency, throughput, memory, task-success, safety, and
  reliability benchmarks still require the versioned hardware protocols and
  raw artifacts defined in the metrics contract.
- Verify the exact published commit and GitHub prerelease after the automated
  quality gates. Repository visibility changes require owner administration access.

## Fresh Linux validation, examples and Docker — 2026-10-09

- A container with no source mounts received a clean V1 source export. The old
  editable package was removed. An isolated core-only environment ran the
  Actions core selection: **89 passed, 4 optional skips**. The simulation
  selection ran **20 passed, 1 optional skip**, including real G1 MuJoCo /
  Spark RL execution. These checks used Linux aarch64, Python 3.10.20 and
  JAX/JAXlib 0.6.2; they do not claim a completed GitHub amd64 run.
- The full CPU regression exposed two obsolete, assertion-free CFS diagnostic
  scripts collected as tests and one PegasusFlow test depending on an
  unpublished local checkout. The scripts were removed. The basis test now
  uses an independent exact Catmull–Rom reference and checks interpolation
  and partition of unity. It ran in the core-only environment: **2 passed,
  3 existing Torch-dependent skips**; the Torch equation tests passed in the
  earlier full environment.
- The corrected broad regression reached 99% with no failures reported in
  completed tests. Host disk exhaustion then disrupted Docker storage and
  prevented retrieval of a final result. This run is **incomplete evidence**.
  A final broad regression result is still required before qualification.
- The final installed wheel passed `pip check`, Twine, import/version and
  configuration checks: **80 experiment and 16 deployment configurations**.
  Both checked-in examples ran outside the source checkout. The 2GO loop
  reached the goal in **48 steps, error 0.099**. The **4096 × 20** projection
  batch passed device, finite-value, bounds and residual assertions; maximum
  residual was **2.24e-8**. These are CPU functional checks, not GPU benchmarks.
- Examples now accept `--device cpu|gpu`. Requested GPU absence fails before
  planner initialization; backend array placement is checked. Five documented
  GPU task configurations passed dry-run validation. No NVIDIA GPU execution
  was performed, and MGA GPU remains deferred.
- The Docker documentation update reuses existing local images. Dockerfiles,
  installers and Compose definitions are preserved. It adds no image rebuild
  requirement or Docker build CI gate. No fresh-image or GPU execution is
  claimed. The prerelease workflow retains package, docs, core and simulation
  checks before publishing distributions and checksums.
