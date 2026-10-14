"""Prepare an owned native Play2Perfect runtime without accepting EULA/starting Kit.

Doctor reads package metadata and performs only a tiny Torch operation. It never
imports isaacsim, isaaclab, isaacsimenvs or the original auto-EULA evaluation code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "ff2fc62873f6c6cd93164123c8358e9ce2d65c9b"
DEFAULT_OUTPUT = ROOT / "results/_development/p3_20261014/results/dexterous/tight_insertion/qualification/play2perfect_native/level_source/seed_0"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2): h.update(chunk)
    return h.hexdigest()


def owned_environment(runtime):
    from scripts.baselines.environment import runtime_environment
    env = runtime_environment(runtime)
    directories = {"UV_CACHE_DIR": "uv-cache", "UV_PYTHON_INSTALL_DIR": "python",
        "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "share", "OMNI_KIT_CACHE_PATH": "kit-cache",
        "CUDA_CACHE_PATH": "cuda-cache", "__GL_SHADER_DISK_CACHE_PATH": "gl-cache",
        "OMNI_CONFIG_PATH": "omniverse-config"}
    for key, relative in directories.items():
        directory = runtime / relative
        directory.mkdir(exist_ok=True)
        env[key] = str(directory)
    # Kit's global tokens otherwise consult ~/.nvidia-omniverse even in portable
    # mode. Its documented config redirects global data/cache/log roots too.
    omni_paths = {"data_root": runtime / "share/omni", "cache_root": runtime / "cache/omni", "logs_root": runtime / "logs/omni"}
    for path in omni_paths.values(): path.mkdir(parents=True, exist_ok=True)
    (runtime / "omniverse-config/omniverse.toml").write_text("[paths]\n" + "".join(
        f"{key} = {json.dumps(str(path))}\n" for key, path in omni_paths.items()))
    env.update(OMNI_KIT_ACCEPT_EULA="NO", UV_LINK_MODE="hardlink", PYTHONNOUSERSITE="1",
               PIP_REQUIRE_VIRTUALENV="true", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


class Budget:
    def __init__(self, runtime, env, args):
        self.runtime, self.env, self.args = runtime, env, args
        previous = runtime / "installation.json"
        prior = json.loads(previous.read_text()) if previous.is_file() else {}
        self.started = time.monotonic() - float(prior.get("wall_seconds", 0))
        self.commands = list(prior.get("commands", []))
        self.peak_bytes = int(prior.get("peak_allocated_runtime_bytes", 0))
        self.invocation_status = "running"

    def check(self):
        # Installers atomically rename/remove cache files while we inspect them.
        # Count allocated blocks once per inode, tolerating only disappearing files.
        allocated, seen = 0, set()
        for directory, _, files in os.walk(self.runtime):
            for path in [Path(directory), *(Path(directory) / name for name in files)]:
                try: item = path.lstat()
                except FileNotFoundError: continue
                if stat.S_ISLNK(item.st_mode) or (item.st_dev, item.st_ino) in seen: continue
                seen.add((item.st_dev, item.st_ino))
                allocated += item.st_blocks * 512
        self.peak_bytes = max(self.peak_bytes, allocated)
        if allocated > self.args.max_runtime_gib * 1024**3:
            raise RuntimeError("Play2Perfect owned runtime exceeded storage budget")
        if shutil.disk_usage(self.runtime).free < self.args.min_free_gib * 1024**3:
            raise RuntimeError("Play2Perfect installation reached reserved free-space floor")
        if time.monotonic() - self.started > self.args.max_seconds:
            raise RuntimeError("Play2Perfect installation exceeded wall-time budget")

    def run(self, name, command, *, cwd=ROOT):
        self.check()
        path = self.runtime / f"{name}.log"
        record = {"stage": name, "argv": list(map(str, command)), "log": str(path), "status": "running"}
        self.commands.append(record)
        self.receipt()
        with path.open("a") as log:
            child = subprocess.Popen(command, cwd=cwd, env=self.env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while child.poll() is None:
                    self.check()
                    time.sleep(2)
                record.update(returncode=child.returncode, status="complete" if child.returncode == 0 else "failed")
                if child.returncode:
                    raise RuntimeError(f"Play2Perfect stage {name} failed; see {path}")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    record.update(returncode=child.returncode, status="terminated_by_budget_or_exception")
                self.receipt()
        print(f"Completed native preparation stage: {name}", flush=True)

    def receipt(self):
        payload = {"schema": "play2perfect-installation-budget-v1", "commands": self.commands,
            "wall_seconds": time.monotonic() - self.started, "peak_allocated_runtime_bytes": self.peak_bytes,
            "limits": {"seconds": self.args.max_seconds, "runtime_gib": self.args.max_runtime_gib,
                       "minimum_free_gib": self.args.min_free_gib},
            "EULA": "not accepted; every child explicitly OMNI_KIT_ACCEPT_EULA=NO",
            "last_invocation_status": self.invocation_status,
            "Kit_started": False}
        (self.runtime / "installation.json").write_text(json.dumps(payload, indent=2) + "\n")


def prepare_assets(runtime, budget):
    from scripts.baselines.sources import LOCK, verify
    source = ROOT / "baselines/Play2Perfect"
    spec = next(item for item in json.loads(LOCK.read_text())["sources"] if item["method"] == "play2perfect")
    verified = verify(spec)
    budget.run("assets_sparse_checkout", ["git", "-C", str(source), "sparse-checkout", "add", "assets"])
    verify(spec)
    paths = subprocess.check_output(["git", "-C", str(source), "ls-tree", "-r", "--name-only", COMMIT, "assets"], text=True).splitlines()
    members = []
    for relative in paths:
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing or symlink native source asset: {relative}")
        expected = subprocess.check_output(["git", "-C", str(source), "rev-parse", f"{COMMIT}:{relative}"], text=True).strip()
        actual = subprocess.check_output(["git", "-C", str(source), "hash-object", str(path)], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"Native asset differs from locked Git blob: {relative}")
        members.append({"path": relative, "bytes": path.stat().st_size, "git_blob": expected, "sha256": sha256(path)})
    import xml.etree.ElementTree as ET
    robot = source / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    urdf = ET.parse(robot).getroot()
    joints = [joint.attrib["name"] for joint in urdf.findall("joint") if joint.attrib.get("type") != "fixed"]
    if len(joints) != 29: raise RuntimeError("Released native robot must retain IIWA7+Sharpa22")
    receipt = {"source": verified, "source_commit": COMMIT, "members": members,
        "asset_files": len(members), "total_bytes": sum(row["bytes"] for row in members),
        "robot": {"name": urdf.attrib["name"], "actuated_joint_count": len(joints), "joint_names": joints},
        "qualification": "public source assets verified; no simulator started"}
    (runtime / "assets_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


def doctor(runtime, env, gpu):
    python = runtime / "venv/bin/python"
    executable = python.resolve()
    if not executable.is_relative_to(runtime): raise RuntimeError("Native CPython must remain inside the owned runtime")
    (runtime / "interpreter.json").write_text(json.dumps({"version": "3.11.14", "executable": str(executable),
        "sha256": sha256(executable), "bootstrap_uv_version": "0.13.1", "managed_install_dir": env["UV_PYTHON_INSTALL_DIR"]}, indent=2) + "\n")
    local = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
    code = '''
import importlib.metadata as m,json,os,sys
assert os.environ.get("OMNI_KIT_ACCEPT_EULA")=="NO"
assert sys.version_info[:2]==(3,11)
names=("torch","torchvision","numpy","isaaclab","isaacsim","typing_extensions","play2perfect","rl-games")
versions={name:m.version(name) for name in names}
assert versions["isaaclab"]=="2.3.2.post1" and versions["isaacsim"].startswith("5.1.0")
assert versions["torch"].startswith("2.7.0") and versions["numpy"]=="1.26.0"
import torch
assert torch.cuda.is_available(),"Tiny Torch CUDA doctor requires available selected GPU"
value=torch.eye(8,device="cuda:0") @ torch.eye(8,device="cuda:0")
torch.cuda.synchronize()
assert torch.equal(value,torch.eye(8,device="cuda:0"))
assert not any(name in sys.modules for name in ("isaacsim","isaaclab","isaacsimenvs"))
print(json.dumps({"python":sys.version,"python_executable":sys.executable,"versions":versions,
"gpu":torch.cuda.get_device_name(0),"CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],
"torch_cuda":torch.version.cuda,"EULA":"not accepted; OMNI_KIT_ACCEPT_EULA=NO",
"scope":"metadata and tiny Torch CUDA only; no Isaac imports/Kit/physics"},indent=2))
'''
    result = subprocess.run([str(python), "-c", code], cwd=ROOT, env=local, text=True, capture_output=True, timeout=60)
    (runtime / "doctor.log").write_text(result.stdout + result.stderr)
    if result.returncode: raise RuntimeError(f"Native metadata/Torch doctor failed: {runtime / 'doctor.log'}")
    report = json.loads(result.stdout)
    (runtime / "doctor.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def persist(runtime, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("installation.json", "assets_manifest.json", "requirements.lock.txt", "doctor.json", "doctor.log", "pip_check.log", "pip_check_deviations.json", "typing_extensions_override.json", "interpreter.json"):
        if (runtime / name).is_file(): shutil.copy2(runtime / name, output / f"native_{name}")
    logs = output / "native_preparation_logs"
    logs.mkdir(exist_ok=True)
    for path in runtime.glob("*.log"): shutil.copy2(path, logs / path.name)
    doctor_file, checks_file = runtime / "doctor.json", runtime / "pip_check_deviations.json"
    doctor_report = json.loads(doctor_file.read_text()) if doctor_file.is_file() else None
    checks = json.loads(checks_file.read_text()) if checks_file.is_file() else None
    installation = json.loads((runtime / "installation.json").read_text())
    assets = json.loads((runtime / "assets_manifest.json").read_text()) if (runtime / "assets_manifest.json").is_file() else None
    expected_versions = {"isaacsim": "5.1.0.0", "isaaclab": "2.3.2.post1", "torch": "2.7.0+cu126"}
    ready = (doctor_report is not None and assets is not None and assets["asset_files"] == 392
             and installation.get("last_invocation_status") == "complete"
             and checks is not None and (checks.get("pip_check_clean", False) or checks.get("expected_author_typing_override_only", False))
             and all(doctor_report["versions"].get(key) == value for key, value in expected_versions.items())
             and (runtime / "requirements.lock.txt").is_file())
    members = [path for path in output.glob("native_*") if path.is_file() and path.name != "native_preparation.json"]
    members.extend(path for path in logs.rglob("*") if path.is_file())
    artifacts = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(members)]
    report = {"schema": "play2perfect-native-preparation-v1",
        "status": "prepared_with_documented_typing_override" if ready else "incomplete",
        "ready_for_user_authorized_EULA_startup": ready, "EULA_accepted": False, "Kit_started": False,
        "native_closed_loop_qualified": False,
        "scope": "Owned packages/interpreter, verified public native assets, metadata and tiny Torch only; no Isaac import or physics qualification",
        "runtime": str(runtime), "source_commit": COMMIT,
        "interpreter_pin": "CPython3.11.14", "bootstrap_uv_pin": "0.13.1",
        "metadata_doctor": doctor_report, "pip_check_deviation": checks,
        "assets": {"files": assets["asset_files"], "bytes": assets["total_bytes"], "robot": assets["robot"]} if assets else None,
        "installation": installation,
        "cache_isolation": {"configured": True, "physical_Kit_token_check_pending": True,
            "configuration": "OMNI_CONFIG_PATH/omniverse.toml plus Kit --portable-root; XDG/TMPDIR/CUDA/GL caches in runtime",
            "USD_conversion": "original Play/assembly tempfile.mkdtemp then usd_work_dir; TMPDIR points inside runtime",
            "source_fixture_fallback": "published fixture URDFs are present; original source tracked contents must remain unchanged"},
        "reviewed_native_launch": {"gpu": 1, "AppLauncher_actor_environment": "cuda:1", "multi_gpu": False,
            "wall_seconds_including_setup": 600, "owned_gpu_mib": 10240, "owned_rss_gib": 20,
            "all_requested_first_episodes_retained": True, "attempt_reuse_forbidden": True,
            "instruction": "Only after explicit caller EULA authorization, run scripts.baselines.play2perfect_native_rollout using this runtime interpreter."},
        "artifacts": artifacts,
        "code": {str(path.relative_to(ROOT)): sha256(path) for path in
                 (ROOT / "scripts/baselines/play2perfect_native_environment.py", ROOT / "scripts/baselines/play2perfect_native_rollout.py",
                  ROOT / "requirements/baselines/play2perfect-native.txt")}}
    (output / "native_preparation.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "assets", "doctor", "clean"))
    parser.add_argument("--name", default="play2perfect_native")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--max-runtime-gib", type=float, default=60)
    parser.add_argument("--min-free-gib", type=float, default=100)
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.max_runtime_gib <= 0 or args.min_free_gib < 0:
        parser.error("invalid installation resource budgets")
    if Path(args.name).name != args.name or args.name in {"", ".", ".."}: parser.error("name must be one path component")
    runtime = ROOT / ".runtime" / args.name
    if runtime.is_symlink() or runtime.parent.is_symlink(): parser.error("runtime must not be a symlink")
    args.output = args.output.resolve()
    if "latex" in args.output.parts: parser.error("native receipts belong outside the paper")
    expected = {"repository": str(ROOT), "name": args.name, "profile": "play2perfect-native-isaacsim5.1", "source_commit": COMMIT}
    marker = runtime / ".owned-play2perfect-native.json"
    if marker.exists():
        if json.loads(marker.read_text()) != expected: parser.error("ownership marker differs")
    elif args.action not in {"create", "assets"} or (runtime.exists() and any(runtime.iterdir())):
        parser.error("refusing to adopt/delete an unowned runtime")
    if args.action == "clean":
        shutil.rmtree(runtime)
        print("Removed owned Play2Perfect runtime; source assets and config-mirrored evidence retained")
        return
    runtime.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(expected, indent=2) + "\n")
    env = owned_environment(runtime)
    budget = Budget(runtime, env, args)
    try:
        if args.action in {"create", "assets"}: prepare_assets(runtime, budget)
        if args.action == "assets":
            budget.invocation_status = "assets_complete"
            return
        if args.action == "create":
            bootstrap = runtime / "bootstrap"
            if not (bootstrap / "bin/python").is_file(): venv.EnvBuilder(with_pip=True).create(bootstrap)
            budget.run("bootstrap_uv", [str(bootstrap / "bin/python"), "-m", "pip", "install", "uv==0.13.1"])
            uv = bootstrap / "bin/uv"
            budget.run("managed_python", [str(uv), "python", "install", "3.11.14"])
            if not (runtime / "venv/bin/python").is_file():
                budget.run("native_venv", [str(uv), "venv", "--python", "3.11.14", "--managed-python", "--seed", str(runtime / "venv")])
            python = runtime / "venv/bin/python"
            executable = python.resolve()
            if not executable.is_relative_to(runtime):
                raise RuntimeError("Native CPython must live inside the owned runtime")
            (runtime / "interpreter.json").write_text(json.dumps({"version": "3.11.14", "executable": str(executable),
                "sha256": sha256(executable), "bootstrap_uv_version": "0.13.1", "managed_install_dir": env["UV_PYTHON_INSTALL_DIR"]}, indent=2) + "\n")
            prefix = [str(uv), "pip", "install", "--python", str(python)]
            budget.run("torch_cu126", prefix + ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0", "--index-url", "https://download.pytorch.org/whl/cu126"])
            # Isaac pins typing_extensions4.12.2; install its closure first, then
            # apply the author's explicitly documented newer tyro/typing override.
            staged = runtime / "native_dependencies_staged.txt"
            staged.write_text("\n".join(line for line in (ROOT / "requirements/baselines/play2perfect-native.txt").read_text().splitlines()
                                       if not line.startswith("tyro==")) + "\n")
            budget.run("native_dependencies", prefix + ["-r", str(staged), "--extra-index-url", "https://pypi.nvidia.com"])
            budget.run("vendored_rl_games", prefix + ["-e", str(ROOT / "baselines/Play2Perfect/rl_games"), "--no-deps"])
            budget.run("source_package", prefix + ["-e", str(ROOT / "baselines/Play2Perfect"), "--no-deps"])
            before = subprocess.check_output([str(python), "-c", "import importlib.metadata as m;print(m.version('typing_extensions'))"], env=env, text=True).strip()
            budget.run("typing_extensions_override", prefix + ["tyro==1.0.16", "typing_extensions==4.15.0"])
            (runtime / "typing_extensions_override.json").write_text(json.dumps({"before": before, "after": "4.15.0", "reason": "author installation guide: tyro NoExtraItems requires >=4.13 after Isaac resolver", "source_file": "docs/isaacsim_installation.md"}, indent=2) + "\n")
            freeze = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"], env=env, text=True)
            (runtime / "requirements.lock.txt").write_text(freeze)
            check = subprocess.run([str(python), "-m", "pip", "check"], env=env, text=True, capture_output=True)
            (runtime / "pip_check.log").write_text(check.stdout + check.stderr)
            if check.returncode:
                lines = [line for line in (check.stdout + check.stderr).splitlines() if line.strip()]
                expected = all(line.lower().startswith("isaacsim-kernel ") and
                    "typing-extensions==4.12.2" in line.lower().replace("_", "-") and "4.15.0" in line for line in lines)
                deviation = {"pip_check_clean": False, "expected_author_typing_override_only": expected,
                             "messages": lines, "reason": "documented author post-Isaac typing_extensions override for tyro NoExtraItems"}
                (runtime / "pip_check_deviations.json").write_text(json.dumps(deviation, indent=2) + "\n")
                if not expected: raise RuntimeError("Native dependency closure has unapproved deviations; see receipt")
            else:
                (runtime / "pip_check_deviations.json").write_text(json.dumps({"pip_check_clean": True, "messages": []}, indent=2) + "\n")
        report = doctor(runtime, env, args.gpu)
        budget.check()
        print(json.dumps(report, indent=2), flush=True)
        budget.invocation_status = "complete"
    finally:
        if budget.invocation_status == "running": budget.invocation_status = "failed"
        budget.receipt()
        persist(runtime, args.output)


if __name__ == "__main__":
    main()
