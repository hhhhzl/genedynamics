"""Resource stops preserve checkpoints; config paths follow existing layout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from genedynamics.experiments.artifacts import ArtifactPolicy, POLICY_ENV, ResourceExhausted, atomic_json, directory_bytes


@pytest.fixture(autouse=True)
def isolate_resource_policy(monkeypatch):
    monkeypatch.delenv(POLICY_ENV, raising=False)


def test_quota_refusal_preserves_published_checkpoint(tmp_path, monkeypatch):
    from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle, load_bundle
    root = tmp_path / 'artifacts'
    save_bundle(root / 'model', {'x': np.array([1.], np.float32)}, {'generation': 1})
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text(json.dumps({'root': str(root), 'max_artifact_bytes': directory_bytes(root) + 1000, 'min_free_bytes': 0}))
    monkeypatch.setenv(POLICY_ENV, str(policy_path))
    with pytest.raises(ResourceExhausted):
        save_bundle(root / 'model', {'x': np.ones(4096, np.float32)}, {'generation': 2})
    arrays, meta = load_bundle(root / 'model')
    assert meta['generation'] == 1
    np.testing.assert_array_equal(arrays['x'], [1.])
    assert not list(root.rglob('.tensors-*'))


def test_free_space_refusal_and_path_escape(tmp_path, monkeypatch):
    from collections import namedtuple
    from genedynamics.experiments import artifacts
    Usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(artifacts.shutil, 'disk_usage', lambda p: Usage(100, 90, 10))
    policy = ArtifactPolicy(str(tmp_path / 'run'), 1000, 10)
    with pytest.raises(ResourceExhausted):
        with policy.write(tmp_path / 'run/data', 1):
            pytest.fail('write must not start')
    with pytest.raises(ValueError, match='outside'):
        with policy.write(tmp_path / 'other/data', 1):
            pytest.fail('path must not escape')


def test_concurrent_writers_share_one_quota(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'root': str(root), 'max_artifact_bytes': 12000, 'min_free_bytes': 0}))
    code = """
import sys, time
from pathlib import Path
from genedynamics.experiments.artifacts import artifact_write, ResourceExhausted
try:
    with artifact_write(sys.argv[1], 8000):
        time.sleep(.2)
        Path(sys.argv[1]).write_bytes(b'x'*8000)
except ResourceExhausted:
    sys.exit(42)
"""
    env = dict(os.environ, **{POLICY_ENV: str(policy)})
    children = [subprocess.Popen([sys.executable, '-c', code, str(root / str(i))], env=env) for i in range(2)]
    assert sorted(c.wait(timeout=10) for c in children) == [0, 42]
    assert directory_bytes(root) == 8000


def test_pilot_config_mirrors_task_method_and_level_seed():
    from genedynamics.experiments.framework.config import ExperimentConfig
    cfg = ExperimentConfig.from_yaml(Path('configs/arm/keyed_insertion_recovery/validation/cross_ebdm.yaml'))
    assert not cfg.validate()
    relative = cfg.output_dir.relative_to(cfg._find_project_root())
    assert str(relative) == 'results/arm/keyed_insertion_recovery/validation/cross_ebdm'
    cfg.use_development_output_root(Path('results/_development/test_protocol'))
    assert cfg.output_dir == cfg._find_project_root() / 'results/_development/test_protocol' / relative
    assert not cfg.metadata['formal_seeds']


def test_timed_out_process_publishes_failure(tmp_path):
    from genedynamics.experiments.process import run_stage
    with pytest.raises(TimeoutError):
        run_stage('timeout', [sys.executable, '-c', 'import time;time.sleep(20)'], tmp_path, timeout=.1)
    status = json.loads((tmp_path / 'timeout.stage.json').read_text())
    assert status['status'] == 'failed' and status['error_type'] == 'TimeoutError'


def test_frozen_map_contract_survives_json_round_trip():
    from genedynamics.experiments.artifacts import canonical_json
    original = {'methods': {'transport': {'scale': 2.5, 'bias': (-.008, .004)}}}
    saved = json.loads(json.dumps(original))
    assert saved == canonical_json(original)
    changed = {'methods': {'transport': {'scale': 1., 'bias': (-.008, .004)}}}
    assert saved != canonical_json(changed)


def test_completed_study_resume_does_not_charge_or_execute_again(tmp_path, monkeypatch):
    """Exercise the actual study orchestrator with a cheap counted physics stub."""
    from types import SimpleNamespace
    from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle
    from genedynamics.learning.cross_embodiment.contact import ContactMechanismSpec
    from scripts.cross_embodiment import run_theory_feasibility as study
    calls = {'probes': 0, 'rollouts': 0}

    def probes(*args, **kwargs):
        calls['probes'] += 5
        return [None] * 5, {'limit': 5, 'attempts': [{'outcome': 'success'}] * 5}

    def rollout(config, mechanism, *, seed, mode, output, **kwargs):
        calls['rollouts'] += 1
        row = dict(success=True, status=1, seed=seed, mode=mode, observed_mode=mode,
                   final_position=[.44, -.2, .580], final_force_rmse=0.,
                   response_signature=[0.] * 24, seconds=1., force_peak=1.)
        save_bundle(output, {'stub': np.ones(1)}, row)
        return row

    class Mapping:
        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(study, 'collect_probes', probes)
    monkeypatch.setattr(study, 'noise_records', lambda records, *a: records)
    monkeypatch.setattr(study, 'rollout', rollout)
    monkeypatch.setattr(study, 'fit_contact_twin', lambda *a: (ContactMechanismSpec(.65, 3), {}))
    monkeypatch.setattr(study, 'validate_twin', lambda *a: {'supported': True})
    monkeypatch.setattr(study.FrozenRegressor, 'load', lambda *a, **kw: None)
    monkeypatch.setattr(study, 'PhaseGaussianPrior', lambda _: SimpleNamespace(regressor=SimpleNamespace(metadata={'tensor_sha256': 'source'})))
    monkeypatch.setattr(study, 'transport', lambda *a, **kw: Mapping())
    monkeypatch.setattr(study.ReferenceTransport, 'load', lambda *a, **kw: Mapping())
    monkeypatch.setattr(study, 'ResearchAudit', lambda: SimpleNamespace(cost_certificate=lambda *a: {}, mode_law=lambda *a: {}))
    argv = ['study', '--output', str(tmp_path / 'study'), '--prior', str(tmp_path / 'prior')]
    monkeypatch.setattr(sys, 'argv', argv)
    study.main()
    first = dict(calls)
    assert first == {'probes': 5, 'rollouts': 180}
    study.main()
    assert calls == first
    monkeypatch.setattr(sys, 'argv', argv + ['--gain', '.8'])
    with pytest.raises(ValueError, match='contract/code mismatch'):
        study.main()
    assert calls == first
