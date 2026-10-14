"""Create, inspect or remove the isolated native FADA MuJoCo/mjwarp runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

from scripts.baselines.environment import runtime_environment

ROOT = Path(__file__).resolve().parents[2]
MJWARP_COMMIT = "09ec1da65ac44bd68703f9507c9c9e8f22ae7c71"


def reuse_wheels(reference, target):
    """Copy immutable wheel files as independent hardlinks, without reference .pth."""
    code = "import sysconfig;print(sysconfig.get_paths()['purelib'])"
    source = Path(subprocess.check_output([str(reference), "-c", code], text=True).strip())
    destination = Path(subprocess.check_output([str(target), "-c", code], text=True).strip())
    existing = {dist.metadata["Name"].lower().replace("_", "-")
                for dist in importlib.metadata.distributions(path=[str(destination)])}
    copied = []
    for distribution in importlib.metadata.distributions(path=[str(source)]):
        name = distribution.metadata["Name"]
        if name.lower().replace("_", "-") in existing:
            continue
        for entry in distribution.files or []:
            original = Path(distribution.locate_file(entry))
            try:
                relative = original.relative_to(source)
            except ValueError:
                continue
            if ".." in relative.parts or not original.is_file():
                continue
            output = destination / relative
            if output.exists():
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(original, output)
            except OSError:
                shutil.copy2(original, output)
        copied.append({"name": name, "version": distribution.version})
    return {"source_python": str(reference), "method": "independent immutable wheel files; no runtime reference .pth",
            "distributions": copied}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "doctor", "clean"))
    parser.add_argument("--name", default="fada_native")
    parser.add_argument("--reuse-python", type=Path)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {"", ".", ".."}:
        parser.error("name must be one directory component")
    runtime = ROOT / ".runtime" / args.name
    if runtime.is_symlink() or runtime.parent.is_symlink():
        parser.error("Runtime must not be a symlink")
    marker = runtime / ".owned-fada-native-environment.json"
    expected = {"repository": str(ROOT), "name": args.name, "profile": "fada-native-mujoco",
                "mujoco_warp_commit": MJWARP_COMMIT}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            parser.error("Native FADA ownership marker differs")
    elif args.action != "create" or (runtime.exists() and any(runtime.iterdir())):
        parser.error("No native FADA ownership marker; refusing to adopt or delete directory")
    if args.action == "clean":
        shutil.rmtree(runtime)
        print("Removed native FADA packages/caches; config-mirrored results and source remain")
        return
    runtime.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(expected, indent=2) + "\n")
    environment = runtime_environment(runtime)
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["WARP_CACHE_PATH"] = str(runtime / "warp-cache")
    environment["PYTHONPATH"] = str(ROOT / "baselines/FADA/src/holosoma") + os.pathsep + str(ROOT)
    python = runtime / "venv/bin/python"
    if args.action == "create":
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(runtime / "venv")
        if args.reuse_python:
            receipt = reuse_wheels(args.reuse_python.resolve(), python)
            (runtime / "wheel-reuse.json").write_text(json.dumps(receipt, indent=2) + "\n")
        commands = [
            [str(python), "-m", "pip", "install", "setuptools>=77", "wheel"],
            [str(python), "-m", "pip", "install", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu124"],
            [str(python), "-m", "pip", "install", "--report", str(runtime / "install-report.json"),
             "-r", str(ROOT / "requirements/baselines/fada-native-mujoco.txt")],
        ]
        for command in commands:
            subprocess.run(command, env=environment, cwd=ROOT, check=True)
        source = runtime / "mujoco_warp"
        if not source.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            "https://github.com/google-deepmind/mujoco_warp.git", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "checkout", MJWARP_COMMIT], check=True)
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != MJWARP_COMMIT:
            raise RuntimeError("Existing native MuJoCo Warp revision differs from the FADA release")
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(source)], env=environment, check=True)
        subprocess.run([str(python), "-m", "pip", "check"], env=environment, check=True)
        frozen = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"], env=environment, text=True)
        (runtime / "requirements.lock.txt").write_text(frozen)
    code = """
import json,sys,torch,mujoco,warp,mujoco_warp
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from genedynamics.solvers.single.fada import FADASolver
assert torch.cuda.is_available(), 'Native FADA CUDA unavailable; no CPU fallback'
assert hasattr(mujoco_warp,'put_model'), 'Native GPU backend API missing'
print(json.dumps({'python':sys.version,'torch':torch.__version__,'mujoco':mujoco.__version__,
                  'warp':warp.__version__,'gpu':torch.cuda.get_device_name(0),
                  'scope':'native dependency/API doctor only; closed loop qualified separately'},indent=2))
"""
    result = subprocess.run([str(python), "-c", code], env=environment, cwd=ROOT, text=True, capture_output=True)
    (runtime / "doctor.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise SystemExit(f"Native FADA doctor failed; see {runtime / 'doctor.log'}")
    print(result.stdout)


if __name__ == "__main__":
    main()
