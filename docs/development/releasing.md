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

- select and add the open-source license;
- confirm the public repository name and URLs;
- approve paper media and citation metadata;
- approve the final release notes and tag.

Tagging and publishing are final external actions and happen only after the
candidate diff and these decisions are reviewed.
