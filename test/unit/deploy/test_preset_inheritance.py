"""Tests verifying the sim → real preset diff is exactly the documented set.

Phase 7's central promise is "switch sim to real by overriding two or
three inner classes". This file enforces that contract by diffing the
flattened ``config_to_dict`` of both presets and asserting that the
non-overridden sections (``runtime``, ``robot``, ``safety``,
``control_hz``, ``sim_dt``, ``max_steps``) are byte-equal across the two
presets.

If somebody adds a real-only knob in a section that should be inherited,
this test fails — which is the only reliable way to keep the contract
honest as the codebase grows.
"""

from __future__ import annotations

import pytest

from genedynamics.deploy.config_schema import config_to_dict
from genedynamics.deploy.presets.g1_corridor_mujoco_sport_mode import (
    G1CorridorMujocoSportModePreset,
)
from genedynamics.deploy.presets.g1_corridor_real_sport_mode import (
    G1CorridorRealSportModePreset,
)


SIM = config_to_dict(G1CorridorMujocoSportModePreset)
REAL = config_to_dict(G1CorridorRealSportModePreset)


# ---------------------------------------------------------------------------
# Inherited (must be identical)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["runtime", "robot", "safety"])
def test_inherited_sections_byte_equal(field):
    assert SIM[field] == REAL[field], (
        f"Section {field!r} differs between sim and real presets — "
        f"this should not happen unless the sim → real contract is changing.\n"
        f"  sim:  {SIM[field]}\n"
        f"  real: {REAL[field]}"
    )


@pytest.mark.parametrize("field", ["control_hz", "sim_dt", "max_steps"])
def test_inherited_scalars_match(field):
    assert SIM[field] == REAL[field], (
        f"Scalar {field!r} differs: sim={SIM[field]} real={REAL[field]}"
    )


# ---------------------------------------------------------------------------
# Overridden (must differ in the right way)
# ---------------------------------------------------------------------------


def test_io_registry_key_swapped():
    assert SIM["io"]["registry_key"] == "io.mujoco"
    assert REAL["io"]["registry_key"] == "io.unitree_g1"


def test_loco_client_registry_key_swapped():
    assert SIM["controller"]["loco_client"]["registry_key"] == "loco_client.spark_rl"
    assert REAL["controller"]["loco_client"]["registry_key"] == "loco_client.real"


def test_real_io_carries_localization_factory():
    """The real preset must declare a localization plugin (Lazy factory)."""
    from genedynamics.deploy.config_schema import Lazy

    loc = REAL["io"].get("localization")
    assert isinstance(loc, Lazy), (
        f"real preset io.localization should be a Lazy(...) factory; got {type(loc)}"
    )


def test_real_controller_uses_lower_gains_than_sim():
    """Real-hardware preset is documented to back off the position gains."""
    assert REAL["controller"]["leg_kp"] < SIM["controller"]["leg_kp"]
    assert REAL["controller"]["leg_kd"] < SIM["controller"]["leg_kd"]


def test_observers_dir_differs():
    sim_dirs = [o["out_dir"] for o in SIM["observers"]]
    real_dirs = [o["out_dir"] for o in REAL["observers"]]
    assert sim_dirs != real_dirs
    assert all("real" in d for d in real_dirs)


# ---------------------------------------------------------------------------
# The diff itself — both as a sanity check and as documentation
# ---------------------------------------------------------------------------


def test_only_documented_sections_differ():
    """No surprise overrides — only io / controller / observers should differ."""
    differing = sorted(k for k in SIM if SIM.get(k) != REAL.get(k))
    expected = sorted(["io", "controller", "observers"])
    assert differing == expected, (
        f"Unexpected sections differ between sim and real presets.\n"
        f"  expected diff: {expected}\n"
        f"  actual diff:   {differing}\n"
        f"If you intentionally added a new section that should differ in real,\n"
        f"add it to the 'expected' list in this test."
    )
