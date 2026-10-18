"""Freeze existing T2/T4 physical/public semantics without rewriting T1/T3."""
from dataclasses import asdict
import hashlib
import json

from .protocol import ROOT,canonical,load_frozen_contract
from .guide_observation import FEATURE_VERSION,CONTEXT_DIM
from .specs import task_spec

VERSION='cross-embodiment-public-task-contract-v1'


def contract_path(task):
    spec=task_spec(task)
    if spec.alias not in {'T2','T4'}:raise ValueError('separate extension for T2/T4 only')
    return ROOT/('configs/quadruped' if spec.alias=='T4' else 'configs/arm')/spec.name/'protocol/task_contract_v1.json'


def current_contract(task):
    spec=task_spec(task);base,base_sha=load_frozen_contract('T3');mobile=spec.alias=='T4'
    if spec.alias not in {'T2','T4'}:raise ValueError('T2/T4 only')
    source=task_spec('T3' if mobile else 'T2',robot='panda');target=task_spec(task,robot='go2_z1' if mobile else 'xarm7')
    return dict(version=VERSION,task=spec.name,alias=spec.alias,
        source=dict(task=source.name,robot='panda',config=asdict(source.config)),
        target=dict(task=target.name,robot=target.config.robot,config=asdict(target.config)),
        initial='jammed' if mobile else 'clear',
        public_schema=spec.describe()['observations'],excluded_policy_fields=spec.describe()['excluded_policy_fields'],
        shared_memory=dict(version=FEATURE_VERSION,dim=CONTEXT_DIM,clock='every20ms observation; phase and source draw every10ticks',
            event='T3 withdrawal/alignment/insertion template; measured TCP pose/wrench and known keyway' if mobile else 'measured contact; x>.455 or y>-.181 guide outlet; public pose/force only'),
        mode_semantics=list(spec.mode_semantics),actions=base['actions'],permissions=base['permissions'],development_budgets=base['development_budgets'],
        inherited_rules_from=dict(path=str((ROOT/'configs/arm/keyed_insertion_recovery/protocol/public_v3.json').relative_to(ROOT)),canonical_sha256=base_sha),
        evaluation=dict(cost='0 iff independent physical success and requested-mode match,1 otherwise; all requests retained',
            deadline_seconds=target.config.timeout_seconds,position_tolerance_m=target.config.position_tolerance,orientation_tolerance_rad=target.config.orientation_tolerance,
            measured_force_hold_N=[target.config.desired_force-2.5,target.config.desired_force+2.5],hold_seconds=.3,
            total_substep_force_limit_N=target.config.force_limit,lateral_substep_force_limit_N=target.config.lateral_force_limit,
            guide_scope='spatial dual-channel guide; current modeled tool/clearance, not draft proposed perturbation range' if not mobile else None,
            body=dict(height_min_m=.15,tilt_max_rad=.5,wbc_residual_max=.01,support_min_after_51_ticks=2,WBC='same host OSQP stance WBC for all methods; no navigation/gait claim') if mobile else None,
            evaluator='genedynamics.tasks.transfer.evaluation.evaluate_episode; unchanged predicates, physical truth used only by evaluator'),
        development_cells=[dict(robot=target.config.robot,servo_gain=1.,delay_steps=0,payload_mass=.5),dict(robot=target.config.robot,servo_gain=.65,delay_steps=3,payload_mass=.8)],
        reset_scope='Go2 task-space noise is0; distinct seeds at same mechanism are not distinct physical initial conditions. Reset diversity and body/motor cells require separate qualification.' if mobile else 'existing0.3mm source/target task reset noise; no expanded draft perturbation claim',
        status='contract_frozen_not_scene_or_method_admission',GPU_physics_qualified=False,
        final_test=False,real_robot_evidence=False)


def load_frozen_contract_extension(task):
    path=contract_path(task);value=json.loads(path.read_text());digest=hashlib.sha256(canonical(value)).hexdigest()
    if digest!=path.with_suffix('.sha256').read_text().strip() or canonical(value)!=canonical(current_contract(task)):
        raise ValueError('extended contract drift; create a new reviewed version')
    return value,digest


def freeze(task):
    path=contract_path(task)
    if path.exists() or path.with_suffix('.sha256').exists():raise FileExistsError('cannot overwrite frozen extension')
    value=current_contract(task);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');path.with_suffix('.sha256').write_text(hashlib.sha256(canonical(value)).hexdigest()+'\n')
    return load_frozen_contract_extension(task)
