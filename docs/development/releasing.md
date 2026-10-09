# Releasing V1

Use the existing `hhhhzl/genedynamics` repository. Prepare changes on
`release/v1-open-source`, then publish from `main`. Follow the
[V1 checklist](../releases/v1_checklist.md).

## Validate the candidate

```bash
python scripts/release/validate_release.py
python -m build
python -m twine check dist/*
pytest test/unit test/algos -m "not performance and not gpu and not docker and not brax and not isaac_lab"
mkdocs build --strict
```

Install the wheel in a fresh virtual environment. Run the Python planning
example, batch projection example and documented CLI dry run from outside the
source checkout. Run the simulator acceptance environment separately. Inspect
wheel and source inventories for credentials, generated results, large files,
excluded research systems and all bundled license texts.

## Promote and publish

1. Push the validated preparation branch and verify its Actions checks.
2. Merge it into `main` through a pull request or a verified fast-forward.
3. Verify the checks on the exact `main` commit and the repository visibility.
4. Tag that commit `v0.1.0` and create a GitHub prerelease using the checked-in
   [release notes](../releases/v0.1.0.md).
5. Attach the wheel, source archive and SHA-256 checksums from that commit.

The owner selected MIT for original framework code. Preserve all bundled
third-party licenses. GitHub Packages and PyPI are optional distribution
channels; neither is required for a GitHub release. Do not imply that a package
or container is published until its registry URL has been verified.

## Source scope and history

The owner requested removal of soft-robot, co-design and 3DGS research from
the V1 branch and release packages, while preserving existing Git history.
Older commits, branches and PRs therefore remain historical records; they are
not part of the V1 supported source surface. Do not rewrite history or create
a separate public repository for this release.

Keep the official Menagerie revision recorded by the V1 Git tree. Initialize
it from the upstream remote and apply the checked-in setup patch. A local
research submodule commit must not replace this fetchable pin.
