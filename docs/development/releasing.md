# Releasing V1

Work from `release/v1-open-source` and follow
[`docs/releases/v1_checklist.md`](../releases/v1_checklist.md).

## Candidate procedure

```bash
python scripts/release/validate_release.py
python -m build
python -m twine check dist/*
pytest test/unit test/algos -m "not performance and not gpu and not docker and not brax and not isaac_lab"
```

Then install the wheel into a clean virtual environment, run the documented CPU
recipe, and run the simulator acceptance environment. Review the source archive
and wheel inventory for credentials, generated results, large files, and
deferred research systems.

## Required owner decisions

- the project owner selected MIT; preserve all bundled third-party notices;
- confirm the public repository name and URLs;
- approve paper media and citation metadata;
- approve the final release notes and tag.

Tagging and publishing are final external actions and happen only after the
candidate diff and these decisions are reviewed.

## Publishable history

The research repository contains earlier work excluded from V1. Deleting files
in the current tree does not remove their historical versions. Prepare the
public repository from an archive of the final V1 commit, initialize an
independent Git repository, and create a new root commit on `main`. Do not copy
the research `.git` directory, clone its history or push its branches and tags.

Preserve the public Menagerie submodule pin recorded by `.gitmodules` and the
V1 Git tree. Initialize it from its upstream remote, then run the checked-in
asset setup script. The research checkout's local submodule revision must not
replace the publishable pin.

Validate a fresh clone of this independent repository. Retain the package
checksums and validation logs, then publish the new public repository and its
`v0.1.0` prerelease with release notes and distribution files.
