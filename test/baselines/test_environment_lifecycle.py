"""Actual subprocess cache isolation and ownership-safe deletion."""
import json
from pathlib import Path
import sys

from scripts.baselines import environment


def test_run_and_clean_preserve_results_and_own_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "ROOT", tmp_path)
    runtime = tmp_path / ".runtime/test"
    (runtime / "venv/bin").mkdir(parents=True)
    (runtime / "venv/bin/python").symlink_to(sys.executable)
    marker = dict(repository=str(tmp_path), name="test", profile="tasks-pytorch", device="cpu")
    (runtime / ".owned-baseline-environment.json").write_text(json.dumps(marker))
    results = tmp_path / "results/keep.json"
    results.parent.mkdir(); results.write_text("preserved")
    receipt = tmp_path / "environment.json"
    code = (f"import json,os,pathlib; pathlib.Path({str(receipt)!r}).write_text(json.dumps({{"
            "k:os.environ[k] for k in ('PIP_CACHE_DIR','TMPDIR','HF_HOME','TORCH_HOME',"
            "'TRITON_CACHE_DIR','PYTHONPYCACHEPREFIX','CUDA_VISIBLE_DEVICES')}))")
    argv = ["environment", "run", "--name", "test", "--profile", "tasks-pytorch", "--device", "cpu",
            "--min-free-gib", "0", "--", "python", "-c", code]
    monkeypatch.setattr(sys, "argv", argv); environment.main()
    actual = json.loads(receipt.read_text())
    assert actual.pop("CUDA_VISIBLE_DEVICES") == ""
    assert all(Path(value).is_relative_to(runtime) for value in actual.values())
    monkeypatch.setattr(sys, "argv", argv[:1] + ["clean"] + argv[2:8])
    environment.main()
    assert not runtime.exists()
    assert results.read_text() == "preserved"
