"""Inventory/fetch author's gated H2S2R native assets without exposing its URL.

Set HUMAN2SIM2ROBOT_ASSET_URL to the base URL issued by the author. This tool
does not submit a form or obtain licensed IsaacGym. It uses the exact archive
routes in the pinned upstream download.py, records first-use SHA256 receipts,
and extracts only bounded, ordinary files in a project-owned runtime directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "baselines/Human2Sim2Robot"
TASKS = ("snackbox_push", "plate_pivotliftrack")
URL_ENV = "HUMAN2SIM2ROBOT_ASSET_URL"


def archive_specs(tasks=TASKS, *, include_raw=False):
    result = [dict(id="assets", route="assets.zip", parent=".", top="assets"),
              dict(id="pretrained_models", route="pretrained_models.zip", parent=".", top="pretrained_models")]
    for task in tasks:
        if task not in TASKS:
            raise ValueError("unknown native task")
        groups = ("human_demo_processed_data", "human_demo_raw_data") if include_raw else ("human_demo_processed_data",)
        for group in groups:
            result.append(dict(id=f"{group}_{task}", route=f"data/{group}/{task}.zip", parent=f"data/{group}", top=task))
    return result


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def owned_root(path, *, create=False):
    path = Path(path).absolute()
    runtime = ROOT / ".runtime"
    if runtime.is_symlink() or path.is_symlink() or runtime not in path.parents:
        raise ValueError("native assets must remain under the project .runtime directory without symlinks")
    if any(parent.is_symlink() for parent in path.parents if parent != ROOT):
        raise ValueError("asset-root ancestors must not be symlinks")
    path = path.resolve()
    if runtime not in path.parents:
        raise ValueError("native asset root cannot traverse out of the project runtime")
    marker = path / ".owned-human2sim2robot-assets.json"
    expected = {"repository": str(ROOT), "method": "human2sim2robot", "schema_version": 1}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError("native asset ownership marker differs")
    elif create:
        if path.exists() and any(path.iterdir()):
            raise ValueError("refusing to adopt a nonempty asset directory")
        path.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(expected, indent=2) + "\n")
    elif path.exists():
        raise ValueError("native asset directory has no ownership marker")
    return path


def safe_extract(archive, destination, *, top, max_unpacked_bytes, max_members=500000):
    """Validate the entire ZIP before extracting into a new private directory."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("ZIP extraction requires a new staging directory")
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) > max_members or sum(item.file_size for item in members) > max_unpacked_bytes:
            raise ValueError("ZIP exceeds the declared member/expanded-size budget")
        seen = set()
        for item in members:
            member = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (member.is_absolute() or ".." in member.parts or "\\" in item.filename
                    or not member.parts or member.parts[0] != top or item.flag_bits & 1
                    or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)))):
                raise ValueError("ZIP contains a path outside its declared tree or a nonordinary member")
            if item.filename in seen:
                raise ValueError("ZIP contains duplicate member paths")
            seen.add(item.filename)
        destination.mkdir()
        hashes = {}
        for item in members:
            target = destination.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as incoming, target.open("xb") as output:
                digest, count = hashlib.sha256(), 0
                while chunk := incoming.read(1024**2):
                    count += len(chunk)
                    if count > item.file_size:
                        raise ValueError("ZIP member size differs from declared size")
                    output.write(chunk); digest.update(chunk)
            if count != item.file_size:
                raise ValueError("truncated ZIP member")
            hashes[str(target.relative_to(destination))] = {"bytes": count, "sha256": digest.hexdigest()}
    return hashes


def verify_archive_receipt(root, spec):
    receipt_path = root / "receipts" / (spec["id"] + ".json")
    if not receipt_path.is_file():
        return {"id": spec["id"], "status": "assets_pending", "route": spec["route"]}
    receipt = json.loads(receipt_path.read_text())
    if receipt["spec"] != spec:
        raise ValueError("native asset receipt protocol differs")
    base = root / spec["parent"]
    for relative, identity in receipt["files"].items():
        path = base / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size != identity["bytes"] or _digest(path) != identity["sha256"]:
            raise ValueError(f"native asset differs from immutable receipt: {spec['id']}")
    return {"id": spec["id"], "status": "asset_receipt_verified", "route": spec["route"],
            "files": len(receipt["files"]), "archive_sha256": receipt["archive_sha256"],
            "publisher_checksum_available": False}


def fetch_archive(root, spec, *, base_url, max_archive_bytes=8 * 1024**3,
                  max_unpacked_bytes=25 * 1024**3, min_free_bytes=20 * 1024**3):
    root = owned_root(root, create=True)
    existing = verify_archive_receipt(root, spec)
    if existing["status"] == "asset_receipt_verified":
        return existing
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.fragment:
        raise ValueError("author asset base URL must use HTTPS")
    url = urllib.parse.urlunsplit(parsed._replace(path=parsed.path.rstrip("/") + "/" + spec["route"]))
    parent = root / spec["parent"]
    parent.mkdir(parents=True, exist_ok=True)
    if (parent / spec["top"]).exists():
        raise ValueError("unreceipted native asset tree exists; refusing to overwrite or adopt it")
    with tempfile.TemporaryDirectory(prefix=".asset-download-", dir=root) as temporary:
        archive = Path(temporary) / "payload.zip"
        digest, count = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(url, timeout=30) as response, archive.open("xb") as output:
                while chunk := response.read(1024**2):
                    count += len(chunk)
                    if count > max_archive_bytes or shutil.disk_usage(root).free < min_free_bytes + len(chunk):
                        raise RuntimeError("native asset download exceeded its disk/size budget")
                    output.write(chunk); digest.update(chunk)
        except urllib.error.HTTPError as error:
            # The author's URL may contain a private access prefix. Never
            # interpolate Request/HTTPError/URLError into a log or receipt.
            raise RuntimeError(f"native asset download returned HTTP status {error.code}; private URL omitted") from None
        except urllib.error.URLError:
            raise RuntimeError("native asset download network failure; private URL omitted") from None
        with zipfile.ZipFile(archive) as zipped:
            expanded = sum(member.file_size for member in zipped.infolist())
        if expanded > max_unpacked_bytes or shutil.disk_usage(root).free < min_free_bytes + expanded:
            raise RuntimeError("native archive expansion exceeds the disk/size budget")
        staging = Path(temporary) / "unpacked"
        files = safe_extract(archive, staging, top=spec["top"], max_unpacked_bytes=max_unpacked_bytes)
        if not files:
            raise ValueError("native archive contains no actual files")
        receipt = {"schema_version": 1, "spec": spec, "archive_bytes": count,
                   "archive_sha256": digest.hexdigest(), "files": files,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "publisher_checksum_available": False,
                   "scope": "first-use SHA256/ZIP integrity receipt; publisher did not supply an independent checksum; no native task qualification"}
        receipts = root / "receipts"; receipts.mkdir(exist_ok=True)
        receipt_path = receipts / (spec["id"] + ".json")
        # Publish verified contents before an immutable receipt. An interrupted
        # publication fails closed as an unreceipted tree on the next invocation.
        (staging / spec["top"]).rename(parent / spec["top"])
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2); stream.write("\n")
    return verify_archive_receipt(root, spec)


def native_manifest(root, *, tasks=TASKS, include_raw=False):
    root = owned_root(root)
    specs = archive_specs(tasks, include_raw=include_raw)
    assets = [verify_archive_receipt(root, spec) for spec in specs]
    workspace = ROOT / ".runtime/human2sim2robot_native/workspace"
    environment = ROOT / ".runtime/human2sim2robot_native/env"
    python = str(environment / "bin/python")
    examples = []
    model_names = {"snackbox_push": "Experiment_snackbox_push_2025-03-28_10-33-38-374687",
                   "plate_pivotliftrack": "Experiment_plate_pivotliftrack_2025-03-28_17-08-54-978176"}
    for task in tasks:
        model = workspace / "pretrained_models" / model_names[task]
        examples.append({"task": task, "stage": "native_checkpoint_closed_loop",
                         "cwd": str(workspace), "argv": [python, "human2sim2robot/sim_training/run.py",
                         "--config-path", str(model), "--config-name", "config", "device_id=0", "graphics_device_id=0",
                         "test=True", "num_envs=1", "headless=True", "train.player.deterministic=False",
                         "task.randomize=False", "task.env.custom.enableDebugViz=False",
                         f"checkpoint={model / 'nn/best.pth'}"],
                         "qualification": "pending; this is a command contract, not an executed score"})
    installation = SOURCE / "docs/installation.md"
    download = SOURCE / "human2sim2robot/utils/download.py"
    return {
        "schema_version": 1, "method": "human2sim2robot", "native_method_qualified": False,
        "status": "assets_obtained_native_environment_pending" if all(row["status"] == "asset_receipt_verified" for row in assets) else "assets_pending",
        "url_environment_variable": URL_ENV, "private_url_provided": bool(os.environ.get(URL_ENV)),
        "source_commit": "894eae2ec3ae39a573b81bd1860d14cc6bdfa6df",
        "source_documents": {str(path.relative_to(ROOT)): _digest(path) for path in (installation, download)},
        "assets": assets, "native_environment": {
            "python": "3.8 (IsaacGym Preview4 supports Python 3.7/3.8)", "torch": "2.1.2+cu118",
            "torchvision": "0.16.2+cu118", "numpy": "1.23.5", "cuda_toolkit": "11.8",
            "fabrics": {"repository": "https://github.com/tylerlum/FABRICS.git", "branch": "Human2Sim2Robot",
                        "commit": "b56c9b1d7c927f5f8b7ca9464a3b5607a3777dae"},
            "curobo": {"repository": "https://github.com/tylerlum/curobo.git", "branch": "Human2Sim2Robot",
                       "commit": "00e8c612790f0b2b5cddb78bc17f861203b46e6d"},
            "isaacgym": "user-supplied licensed IsaacGym_Preview_4_Package.tar.gz; not downloaded by this tool",
            "runtime": str(environment), "workspace": str(workspace), "installed": False},
        "ordered_stages": [
            {"stage": "author_resources", "entrypoint": "python -m scripts.baselines.human2sim2robot_native_assets download --include-raw", "executed": False},
            {"stage": "isolated_native_environment", "requirements": "Python3.8/Torch2.1.2/cu118, pinned FABRICS/cuRobo, licensed IsaacGym Preview4", "executed": False},
            {"stage": "native_workspace", "requirements": "pinned source checkout inside project runtime; assets/data/pretrained_models attached from receipt-verified cache", "executed": False},
            {"stage": "human_demo_processing", "requirements": "native raw RGBD/hand/object processing; or separately disclose reuse of author processed trajectories", "executed": False},
            {"stage": "digital_twin_and_exploration_initialization", "requirements": "native robot/object assets, retargeted_robot.npz and original initialization/controller retained", "executed": False},
            {"stage": "checkpoint_closed_loop", "commands": examples, "executed": False},
            {"stage": "native_training_resume", "requirements": "native experiment config, reward, FABRICS and initialization; bounded PPO update/checkpoint/RNG test, then native quantitative reproduction", "executed": False}],
        "scope": "resource inventory and source-derived native command contract; adapted project training and source-core parity do not satisfy native qualification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "download", "manifest", "clean"))
    parser.add_argument("--root", type=Path, default=ROOT / ".runtime/human2sim2robot_native_assets")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = owned_root(args.root, create=args.action == "download")
    if args.action == "clean":
        if not (root / ".owned-human2sim2robot-assets.json").is_file():
            parser.error("no owned native asset cache to clean")
        shutil.rmtree(root)
        print("Removed owned native asset cache; experiment evidence is preserved")
        return
    if args.action == "download":
        url = os.environ.get(URL_ENV)
        if not url:
            parser.error(f"author-issued base URL is required through {URL_ENV}; no form is submitted automatically")
        for spec in archive_specs(args.tasks, include_raw=args.include_raw):
            fetch_archive(root, spec, base_url=url)
    report = native_manifest(root, tasks=args.tasks, include_raw=args.include_raw)
    if args.output:
        destination = args.output.resolve()
        if ROOT / "latex" in destination.parents:
            parser.error("native qualification manifests belong outside latex/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
