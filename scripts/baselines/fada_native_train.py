"""Unmodified official FADA trainer with explicit mjwarp snapshot compatibility.

The module-level installation also runs in multiprocessing's spawned shadow worker.
"""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("FADA_NATIVE_SOURCE", str(ROOT / "baselines/FADA")))
sys.path.insert(0, str(SOURCE / "src/holosoma"))

from genedynamics.learning.baselines.fada.native_compat import install_mjwarp_snapshot_compatibility

COMPATIBILITY = install_mjwarp_snapshot_compatibility()

from holosoma.fada.planner_idm.train import main

if __name__ == "__main__":
    print("Native FADA compatibility:", COMPATIBILITY, flush=True)
    main()
