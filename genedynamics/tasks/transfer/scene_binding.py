"""Immutable physical/controller/evaluator binding beside the semantic contract."""
import hashlib
import json
from dataclasses import asdict

from .protocol import ROOT, canonical, contract_path, load_frozen_contract


SCENE_BINDING_VERSION = 4


def current_scene_binding(task):
    import mujoco
    from genedynamics.envs.adapters.transfer_task import instantiate_task
    from genedynamics.tasks.transfer.runtime import MechanismSpec, SkillSpec
    roots = (
        ROOT / "genedynamics/envs/domains/manipulation/transfer_workcell",
        ROOT / "genedynamics/tasks/transfer",
    )
    paths = {path for root in roots for path in root.rglob("*.py")}
    paths.update(ROOT / name for name in (
        "genedynamics/core/control/transfer.py",
        "genedynamics/envs/adapters/transfer_task.py",
        "genedynamics/envs/adapters/acp_transfer.py",
        "genedynamics/experiments/plugins/environments/transfer_tasks.py",
        "genedynamics/experiments/plugins/metrics/transfer.py",
        "genedynamics/solvers/single/acp/backends/compliance.py",
        "genedynamics/solvers/single/acp/backends/spatial.py",
    ))
    cells = {}
    for name, robot, mechanism in (("source","panda",MechanismSpec()),("nominal","xarm7",MechanismSpec()),
                                    ("derated","xarm7",MechanismSpec(.65,3,.8))):
        scene = instantiate_task(task, robot=robot, mechanism_spec=mechanism).runner.describe()
        cells[name] = dict(robot=robot,mechanism=asdict(mechanism),scene=scene)
    return dict(contract_sha256=load_frozen_contract(task)[1], mujoco_version=mujoco.__version__,cells=cells,
        skill_box_and_rate_limits=asdict(SkillSpec()),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


def scene_binding_path(task):
    return contract_path(task).with_name(f"public_v{SCENE_BINDING_VERSION}_scene.json")


def freeze_scene_binding(task):
    path = scene_binding_path(task)
    if path.exists() or path.with_suffix(".sha256").exists(): raise FileExistsError("scene binding cannot be overwritten")
    value = current_scene_binding(task)
    path.write_text(json.dumps(value,indent=2)+"\n")
    path.with_suffix(".sha256").write_text(hashlib.sha256(canonical(value)).hexdigest()+"\n")
    return load_scene_binding(task)


def load_scene_binding(task):
    path = scene_binding_path(task)
    value = json.loads(path.read_text()); digest = hashlib.sha256(canonical(value)).hexdigest()
    if path.with_suffix(".sha256").read_text().strip() != digest: raise ValueError("scene binding SHA changed")
    if canonical(value) != canonical(current_scene_binding(task)):
        raise ValueError("model/controller/evaluator/public-memory drift requires a new scene binding version")
    return value, digest
