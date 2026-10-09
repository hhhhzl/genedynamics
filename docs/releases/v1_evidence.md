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

The CPU Docker regression produced:

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

The final release-candidate wheel has:

| Property | Value |
| --- | --- |
| Filename | `genedynamics-0.1.0-py3-none-any.whl` |
| Size | 12,160,185 bytes |
| Files | 789 |
| SHA-256 | `75673cea6b1fa3bd8c4d0352fc1d2c2a1fc95b2222d8e4362b3becd91bc13f48` |
| Required package assets missing | 0 |
| Forbidden V1 paths | 0 |
| Twine metadata check | Passed |

The wheel and all core dependencies were installed into a new Python 3.12
virtual environment on macOS arm64. `pip check`, package/version import, JAX CPU
backend discovery, optional D3IL plugin import, and the documented planar
dry-run all passed from outside the repository checkout.

After the README visual refresh, the wheel was rebuilt and its metadata passed
Twine again. Comparing the archives showed changes only in `METADATA` and
`RECORD`; all runtime/package files are byte-identical to the CPU-tested wheel.

The source and wheel inventory is stored in
`release/v1_inventory.json`. The core dependency license closure is stored in
`release/v1_dependency_licenses.json`.

## Static and documentation evidence

- MkDocs built in strict mode.
- 1,017 release Python files parsed with zero syntax errors.
- `git diff --check` reported no whitespace errors.
- Credential pattern scanning reported zero AWS, GitHub, OpenAI, Slack, or
  private-key signatures.
- The current source inventory reports three files above 10 MiB: two D3IL
  camera meshes and the homepage GIF. None is included in the wheel.
- The 20-package core dependency closure reported no detected strong-copyleft
  license.

## Documentation refresh — 2026-10-09

- The homepage now uses 41 original clips in five groups: MDOC, MD-COAS, 2GO,
  MGA and hardware. Titles, subtitles and rulers were cropped from the inputs;
  the camera composition adds no text. All five groups and camera views were
  visually reviewed.
- The 24-second GIF is 896 × 576 at 6 fps (19.52 MiB). The same composition is
  available as a 1344 × 864, 12 fps MP4 (6.11 MiB). This is presentation playback,
  not a runtime or latency measurement.
- All 41 input sizes and SHA-256 checksums matched
  `docs/assets/showcase_sources/v2/manifest.json`. The exports regenerate from
  these checked-in inputs without the original research folders or simulators.
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
  learning/task/execution palette. The main demonstration media is unchanged.
- The documentation built in strict mode. README links, anchors and Python
  syntax passed static checks. Training-guide commands were checked against
  their parsers; no new training runs were performed. Runtime source, task
  configurations and tests are unchanged by this refresh; the earlier
  regression remains the runtime evidence.


## Open gates

- The project owner must select the project license; the project entry remains
  `UNKNOWN` in the license inventory until then.
- Publication-grade latency, throughput, memory, task-success, safety, and
  reliability benchmarks still require the versioned hardware protocols and
  raw artifacts defined in the metrics contract.
- Tagging and publication require a clean-checkout CI run and explicit owner
  approval.
