# V1 release checklist

The checklist is ordered so each phase produces evidence required by the next.

## 1. Freeze scope

- [x] Create `release/v1-open-source` from the current development head.
- [x] Record included algorithms, tasks, and deferred systems.
- [x] Disconnect soft-robot, co-design, and 3DGS plugins from default imports.
- [x] Remove soft-robot, co-design, and 3DGS source, configurations, tests,
  scripts, docs, assets, and default registrations after dependency review.
- [x] Capture a machine-readable compatibility matrix.

## 2. Make installation reliable

- [x] Fix Python package discovery and verify wheel contents.
- [x] Split core, simulator, Torch integration, visualization, and development
  dependencies into installable extras.
- [x] Define supported Python, JAX, operating system, accelerator, and simulator
  versions.
- [x] Validate clean CPU installation from the built wheel.

## 3. Establish release quality gates

- [x] Delete obsolete or invalid tests and merge unique assertions into current
  tests.
- [x] Separate unit, integration, accelerator, simulator, and benchmark suites.
- [x] Convert script-shaped tests into collected tests or explicit CI commands.
- [x] Add package-build, wheel-install, config-validation, and CPU-smoke CI jobs.
- [x] Record optional GPU and simulator jobs without making unsupported claims.

## 4. Stabilize the user surface

- [x] Define the supported Python API and command-line entry points.
- [x] Add clear errors for unavailable optional integrations.
- [x] Validate all included YAML configurations against plugin registrations.
- [x] Add one minimal and one robot recipe with pinned expected outputs.

## 5. Rewrite product documentation

- [x] Replace the README with a feature-first product story, quickstart, task
  gallery, architecture, compatibility matrix, benchmarks, and roadmap.
- [x] Rebuild docs around installation, concepts, recipes, integrations,
  operations, reference, and development.
- [x] Move research logs and internal records outside the user guide.
- [x] Reuse approved paper GIFs and videos with captions and repository-safe
  assets.

## 6. Prove industrial readiness

- [ ] Publish reproducibility, latency, throughput, compilation, memory, success
  rate, constraint-violation, and reliability measurements where applicable.
- [ ] Run each benchmark under a versioned protocol and retain raw artifacts.
- [x] Document failure behavior, device fallback, logging, and saved-run schema.

## 7. Prepare and publish the release candidate

- [ ] Complete license, citation, contribution, security, and conduct policies.
  Citation, contribution, security, and conduct files are complete; the owner
  license selection remains open.
- [x] Generate an inventory for source, wheel, and documentation artifacts.
- [x] Run secret, large-file, dependency, and license scans. The core dependency
  closure has no detected strong-copyleft package; the project itself remains
  `UNKNOWN` until `LICENSE` is selected.
- [ ] Build the release candidate from a clean checkout and run the release gate.
- [ ] Review the final diff, then tag and publish only after explicit approval.
