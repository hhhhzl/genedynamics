"""Create, inspect or remove one project-owned research environment.

Run with system Python. CUDA, pip cache, temporary files and compiled artifacts
live in .runtime/<name>; experiment results are never removed by this tool.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[2]


def size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def environment(runtime):
    env = dict(os.environ)
    env.update(PIP_CACHE_DIR=str(runtime / "pip-cache"), TMPDIR=str(runtime / "tmp"),
               JAX_COMPILATION_CACHE_DIR=str(runtime / "jax-cache"),
               PYTHONPYCACHEPREFIX=str(runtime / "pycache"),
               MPLCONFIGDIR=str(runtime / "matplotlib"),
               CROSS_EMBODIMENT_NATIVE_LIBRARY=str(runtime / "native/libcross_embodiment.so"),
               PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONUNBUFFERED="1")
    return env


def bounded_run(command, runtime, env, max_bytes, min_free):
    with (runtime / "environment.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                if size(runtime) > max_bytes or shutil.disk_usage(runtime).free < min_free:
                    raise RuntimeError("environment resource_exhausted; see environment.log")
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"environment command failed ({child.returncode}); see environment.log")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "doctor", "clean"))
    parser.add_argument("--name", default="cross_ebdm")
    parser.add_argument("--torch-cpu", action="store_true")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--max-runtime-gib", type=float, default=30)
    parser.add_argument("--min-free-gib", type=float, default=20)
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be one path component")
    runtime = ROOT / ".runtime" / args.name
    if runtime.is_symlink():
        parser.error("runtime must not be a symlink")
    marker = runtime / ".owned-environment.json"
    python = runtime / "venv/bin/python"
    if args.action == "clean":
        if not marker.exists() or json.loads(marker.read_text()).get("repository") != str(ROOT):
            parser.error("refusing to remove a directory without this repository's ownership marker")
        shutil.rmtree(runtime)
        print(f"Removed {runtime}; experiment results preserved")
        return
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("pip-cache", "tmp", "jax-cache", "native", "matplotlib"):
        (runtime / name).mkdir(exist_ok=True)
    if marker.exists() and json.loads(marker.read_text()).get("repository") != str(ROOT):
        parser.error("environment belongs to another repository")
    marker.write_text(json.dumps({"repository": str(ROOT), "name": args.name}) + "\n")
    env = environment(runtime)
    if args.action == "create":
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(runtime / "venv")
        max_bytes, min_free = int(args.max_runtime_gib * 1024**3), int(args.min_free_gib * 1024**3)
        install = [str(python), "-m", "pip", "install", "--report", str(runtime / "install-report.json"),
                   "-r", str(ROOT / "requirements/cross-embodiment.txt"), "-e", str(ROOT) + "[test]"]
        bounded_run(install, runtime, env, max_bytes, min_free)
        if args.torch_cpu:
            bounded_run([str(python), "-m", "pip", "install", "--report", str(runtime / "torch-install-report.json"),
                         "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu"],
                        runtime, env, max_bytes, min_free)
        bounded_run([str(python), "-m", "pip", "check"], runtime, env, max_bytes, min_free)
        frozen = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"], cwd=ROOT, env=env, text=True)
        (runtime / "requirements.lock.txt").write_text(frozen)
        subprocess.run([str(python), "-m", "scripts.cross_embodiment.build_native", "--output",
                        str(runtime / "native/libcross_embodiment.so")], cwd=ROOT, env=env, check=True)
    if not python.exists():
        parser.error("environment missing; run create first")
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS="cuda",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", MUJOCO_GL="egl")
    code = """
import importlib.metadata as m, json, platform, sys
import jax, jax.numpy as jnp, mujoco
devices=jax.devices('gpu')
value=jax.jit(lambda x:x@x)(jnp.eye(32))
value.block_until_ready()
assert value.device.platform=='gpu'
report={'python':sys.version,'platform':platform.platform(),'devices':list(map(str,devices)),
        'versions':{name:m.version(name) for name in ('jax','jaxlib','mujoco','mujoco-mjx','numpy','scipy','osqp')},
        'scope':'CUDA numerical smoke; research physics GPU qualification not performed'}
print(json.dumps(report,indent=2))
"""
    result = subprocess.run([str(python), "-c", code], cwd=ROOT, env=env, text=True, capture_output=True)
    (runtime / "doctor.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise SystemExit(f"CUDA smoke failed; see {runtime / 'doctor.log'}")
    report = json.loads(result.stdout)
    report["gpu_inventory"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv"], text=True)
    report["runtime_bytes"] = size(runtime)
    (runtime / "doctor.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
