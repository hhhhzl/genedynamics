"""Create, inspect or remove a project-owned PyTorch source-parity runtime.

This profile deliberately excludes native simulators/hardware and pretrained
vision assets. Its package freeze and pip reports record the resolved closure.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

from scripts.research.environment import bounded_run, size

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "core-pytorch": ROOT / "requirements/baselines/core-pytorch.txt",
    "tasks-pytorch": ROOT / "requirements/baselines/tasks-pytorch.txt",
}


def runtime_environment(runtime: Path) -> dict:
    env = dict(os.environ)
    local = {
        "PIP_CACHE_DIR": "pip-cache", "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache",
        "HF_HOME": "huggingface", "TORCH_HOME": "torch", "TRITON_CACHE_DIR": "triton",
        "TORCH_EXTENSIONS_DIR": "torch-extensions", "MPLCONFIGDIR": "matplotlib",
        "PYTHONPYCACHEPREFIX": "pycache", "JAX_COMPILATION_CACHE_DIR": "jax-cache",
    }
    for key, directory in local.items():
        path = runtime / directory
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    env.update(PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "doctor", "run", "clean"))
    parser.add_argument("--name", default="baseline_core_torch")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="core-pytorch")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-runtime-gib", type=float, default=25)
    parser.add_argument("--min-free-gib", type=float, default=20)
    argv = sys.argv[1:]
    separator = argv.index("--") if "--" in argv else len(argv)
    command = argv[separator + 1:]
    args = parser.parse_args(argv[:separator])
    if args.action == "run" and not command:
        parser.error("run requires -- python -m <module> [arguments]")
    if args.action != "run" and command:
        parser.error("a command is supported only by run")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be one path component")
    base = ROOT / ".runtime"
    runtime = base / args.name
    if base.is_symlink() or runtime.is_symlink():
        parser.error("runtime and its base must not be symlinks")
    marker = runtime / ".owned-baseline-environment.json"
    expected = {"repository": str(ROOT), "name": args.name,
                "profile": args.profile, "device": args.device}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            parser.error("runtime ownership/profile/device differs; choose another name")
    elif args.action != "create" or (runtime.exists() and any(runtime.iterdir())):
        parser.error("no matching baseline ownership marker; refusing to adopt or remove directory")
    if args.action == "clean":
        shutil.rmtree(runtime)
        print(f"Removed {runtime}; sources and experiment results preserved")
        return
    runtime.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(expected, indent=2) + "\n")
    env = runtime_environment(runtime)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu if args.device == "cuda" else ""
    python = runtime / "venv/bin/python"
    if args.action == "create":
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(runtime / "venv")
        max_bytes = int(args.max_runtime_gib * 1024**3)
        min_free = int(args.min_free_gib * 1024**3)
        index = "cu124" if args.device == "cuda" else "cpu"
        commands = [
            [str(python), "-m", "pip", "install", "--report", str(runtime / "torch-install-report.json"),
             "torch==2.5.1", "--index-url", f"https://download.pytorch.org/whl/{index}"],
            [str(python), "-m", "pip", "install", "--report", str(runtime / "install-report.json"),
             "-r", str(PROFILES[args.profile]), "-e", str(ROOT), "--no-build-isolation"],
            [str(python), "-m", "pip", "check"],
        ]
        # Editable metadata uses modern setuptools; keep the build tools local.
        bounded_run([str(python), "-m", "pip", "install", "setuptools>=77", "wheel"],
                    runtime, env, max_bytes, min_free)
        for command in commands:
            bounded_run(command, runtime, env, max_bytes, min_free)
        frozen = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"], env=env, text=True)
        (runtime / "requirements.lock.txt").write_text(frozen)
    if not python.exists():
        parser.error("venv missing; run create first")
    if args.action == "run":
        if command[0] in {"python", "python3"}:
            command[0] = str(python)
        bounded_run(command, runtime, env, int(args.max_runtime_gib * 1024**3),
                    int(args.min_free_gib * 1024**3))
        print(f"Completed in owned runtime; process output: {runtime / 'environment.log'}")
        return
    code = r'''
import importlib.metadata as m, json, platform, sys, torch
requested = sys.argv[1]
if requested == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but unavailable; CPU fallback is forbidden")
device = torch.device(requested)
torch.manual_seed(0)
model = torch.nn.Linear(16, 8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(4,16, device=device)
loss = model(x).square().mean()
loss.backward()
optimizer.step()
if requested == "cuda": torch.cuda.synchronize()
assert next(model.parameters()).device.type == requested
report = {"python":sys.version,"platform":platform.platform(),"device":str(next(model.parameters()).device),
          "torch_cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name() if requested=="cuda" else None,
          "loss":float(loss.detach().cpu()),
          "versions":{name:m.version(name) for name in ("torch","jax","jaxlib","numpy","scipy","diffusers","einops","zarr")},
          "scope":"Torch numerical update smoke; no native simulator or task physics qualification"}
print(json.dumps(report,indent=2))
'''
    result = subprocess.run([str(python), "-c", code, args.device], cwd=ROOT, env=env,
                            text=True, capture_output=True)
    (runtime / "doctor.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise SystemExit(f"Torch doctor failed; see {runtime / 'doctor.log'}")
    report = json.loads(result.stdout)
    report["runtime_bytes"] = size(runtime)
    (runtime / "doctor.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
