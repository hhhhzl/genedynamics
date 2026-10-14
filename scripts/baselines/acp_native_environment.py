"""Create/remove the independent native ACP profile including vision and dataset dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.baselines.environment import ROOT, runtime_environment
from scripts.research.environment import bounded_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "doctor", "clean"))
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    runtime = ROOT / ".runtime/acp_native"
    python = runtime / "venv/bin/python"
    if args.action == "clean":
        subprocess.run([sys.executable, "-m", "scripts.baselines.environment", "clean",
                        "--name", "acp_native", "--device", "cuda"], check=True)
        return
    if args.action == "create":
        subprocess.run([sys.executable, "-m", "scripts.baselines.environment", "create",
                        "--name", "acp_native", "--device", "cuda", "--gpu", args.gpu,
                        "--max-runtime-gib", "35", "--min-free-gib", "40"], check=True)
    if not python.exists():
        parser.error("native ACP runtime is missing; use create")
    env = runtime_environment(runtime)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.action == "create":
        commands = [
            [str(python), "-m", "pip", "install", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu124"],
            [str(python), "-m", "pip", "install", "--report", str(runtime/"native-install-report.json"),
             "-r", str(ROOT/"requirements/baselines/acp-native.txt")],
            [str(python), "-m", "pip", "check"],
        ]
        for command in commands:
            bounded_run(command, runtime, env, 35*1024**3, 40*1024**3)
    code = "import torch,torchvision,timm,vit_pytorch,imagecodecs,hydra,dill,zarr; import json; print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name()})); assert torch.cuda.is_available()"
    result = subprocess.run([str(python), "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    (runtime/"native-doctor.log").write_text(result.stdout+result.stderr)
    if result.returncode:
        raise SystemExit(f"native ACP doctor failed: {runtime/'native-doctor.log'}")
    report = json.loads(result.stdout)
    (runtime/"native-doctor.json").write_text(json.dumps(report,indent=2)+"\n")
    (runtime/"native-requirements.lock.txt").write_text(subprocess.check_output(
        [str(python),"-m","pip","freeze","--all"], env=env,text=True))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
