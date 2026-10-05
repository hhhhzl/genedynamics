# V1 validation evidence

This page records the release-candidate checks run on 2026-10-05 from branch
`release/v1-open-source`. It is validation evidence, not a performance claim.

## Release surface

- 80 experiment configurations and 16 deployment configurations validated.
- 16 method registrations and 7 published environment families matched the
  machine-readable V1 catalogue.
- Soft-robot, co-design, 3DGS, MBD3D, MRMFMBD, and morphology paths were absent
  from the source and wheel inventories.
- Humanoid push-to-line and MGA GPU remain outside the readiness claim.

## Test evidence

The CPU Docker regression selected 742 portable unit/integration cases:

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
| Size | 12,157,082 bytes |
| Files | 789 |
| SHA-256 | `5d3baeac60ee940f7ea294aee83d9b62dfb30c5f30de265f108eec0e60be5a65` |
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
- The source inventory reports two files above 10 MiB, both D3IL camera mesh
  assets; neither is included in the wheel.
- The 20-package core dependency closure reported no detected strong-copyleft
  license.
- The README's six-task motion wall includes all three original GIFs. Its
  12-second GIF is below 10 MiB; the full-resolution MP4 is below 2 MiB.
  Seven input checksums and their render origins are recorded in
  `docs/assets/showcase_sources/manifest.json`. The compositor regenerates the
  exports without the private result tree or a simulator.

## Open gates

- The project owner must select the project license; the project entry remains
  `UNKNOWN` in the license inventory until then.
- Publication-grade latency, throughput, memory, task-success, safety, and
  reliability benchmarks still require the versioned hardware protocols and
  raw artifacts defined in the metrics contract.
- Tagging and publication require a clean-checkout CI run and explicit owner
  approval.
