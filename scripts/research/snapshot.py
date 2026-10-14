"""Preserve a source overlay and environment evidence before deleting runtime."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from genedynamics.experiments.artifacts import POLICY_ENV, artifact_write, atomic_json
from scripts.research.environment import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=ROOT / '.runtime/cross_ebdm')
    parser.add_argument('--policy', type=Path)
    args = parser.parse_args()
    if args.policy:
        os.environ[POLICY_ENV] = str(args.policy.resolve())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    allowed = {'.py', '.cpp', '.h', '.yaml', '.toml', '.txt', '.tex', '.bib'}
    bases = ['genedynamics', 'scripts', 'test', 'configs', 'requirements', 'latex/cross_ebdm']
    sources = {p for base in bases for p in (ROOT / base).rglob('*')
               if p.is_file() and not p.is_symlink() and p.suffix in allowed}
    sources.update(ROOT / name for name in ('pyproject.toml', 'setup.py', '.gitignore', '.gitmodules', 'README.md'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    reserve = sum(p.stat().st_size + 4096 for p in sources) + 65536
    destination = out / 'source-overlay.zip'
    with artifact_write(destination, reserve):
        fd, temporary = tempfile.mkstemp(dir=out, prefix='.snapshot-')
        os.close(fd)
        try:
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(sources):
                    archive.write(path, str(path.relative_to(ROOT)))
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    files = ['doctor.json', 'install-report.json', 'torch-install-report.json',
             'requirements.lock.txt', 'native/libcross_embodiment.so.json']
    environment = {}
    for name in files:
        source = args.runtime / name
        if not source.exists():
            raise FileNotFoundError(f'environment evidence missing: {source}')
        payload = source.read_bytes()
        target = out / Path(name).name
        with artifact_write(target, len(payload) + 4096):
            fd, temporary = tempfile.mkstemp(dir=out, prefix='.environment-')
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(payload)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        environment[name] = hashlib.sha256(payload).hexdigest()
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'base_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'sources': hashes, 'source_zip_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
              'environment': environment,
              'restore': 'checkout base_commit, overlay source ZIP, initialize pinned assets, create isolated environment'}
    atomic_json(out / 'snapshot.json', report)
    print(json.dumps({'files': len(hashes), 'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
