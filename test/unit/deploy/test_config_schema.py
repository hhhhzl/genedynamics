"""Unit tests for the Phase 9 nested-class config schema and registries."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pytest

from genedynamics.deploy import registries
from genedynamics.deploy.config_schema import (
    BuiltComponents,
    ComponentConfig,
    DeployConfig,
    build_components,
    config_to_dict,
    initialize_class,
)


# ---------------------------------------------------------------------------
# config_to_dict
# ---------------------------------------------------------------------------


class _Outer:
    a = 1
    b = "hello"

    class inner:
        x = 3.14
        y = (1, 2, 3)


def test_config_to_dict_walks_nested_classes():
    out = config_to_dict(_Outer)
    assert out["a"] == 1
    assert out["b"] == "hello"
    assert out["inner"]["x"] == pytest.approx(3.14)
    assert out["inner"]["y"] == (1, 2, 3)


def test_config_to_dict_subclass_overrides_win():
    class Child(_Outer):
        a = 99

    out = config_to_dict(Child)
    assert out["a"] == 99
    assert out["b"] == "hello"


def test_config_to_dict_handles_dict_input():
    assert config_to_dict({"x": 1, "y": [1, 2]}) == {"x": 1, "y": [1, 2]}


def test_config_to_dict_skips_methods():
    class HasMethod:
        a = 1

        def some_method(self):
            return None

    out = config_to_dict(HasMethod)
    assert "some_method" not in out
    assert out["a"] == 1


# ---------------------------------------------------------------------------
# initialize_class
# ---------------------------------------------------------------------------


class _DummyComponent:
    """Test component matching `controller.dummy`."""

    def __init__(self, *, kp: float = 1.0, kd: float = 0.5, name: str = "x") -> None:
        self.kp = kp
        self.kd = kd
        self.name = name


class _DummyWithConfig:
    """Test component matching `controller.dummy_grouped`."""

    def __init__(self, *, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}


@pytest.fixture(autouse=True)
def _register_dummies():
    registries.controller_registry.register("dummy", _DummyComponent)
    registries.controller_registry.register("dummy_grouped", _DummyWithConfig)
    yield


def test_initialize_class_resolves_and_constructs():
    class node(ComponentConfig):
        registry_key = "controller.dummy"
        kp = 12.0
        kd = 1.5
        name = "test"

    obj = initialize_class(node)
    assert isinstance(obj, _DummyComponent)
    assert obj.kp == 12.0
    assert obj.kd == 1.5
    assert obj.name == "test"


def test_initialize_class_extra_kwargs_take_precedence():
    class node(ComponentConfig):
        registry_key = "controller.dummy"
        kp = 1.0

    obj = initialize_class(node, extra_kwargs={"kp": 999.0})
    assert obj.kp == 999.0


def test_initialize_class_bundles_unmatched_kwargs_under_config():
    class node(ComponentConfig):
        registry_key = "controller.dummy_grouped"
        name = "grouped"

        class section_a:
            x = 1
            y = 2

        class section_b:
            z = 3

    obj = initialize_class(node)
    assert isinstance(obj, _DummyWithConfig)
    assert obj.name == "grouped"
    assert obj.config["section_a"] == {"x": 1, "y": 2}
    assert obj.config["section_b"] == {"z": 3}


def test_initialize_class_missing_registry_key_raises():
    class node(ComponentConfig):
        kp = 1.0

    with pytest.raises(KeyError):
        initialize_class(node)


def test_initialize_class_unknown_key_raises():
    class node(ComponentConfig):
        registry_key = "controller.nonexistent"

    with pytest.raises(KeyError):
        initialize_class(node)


# ---------------------------------------------------------------------------
# Registries default registration
# ---------------------------------------------------------------------------


def test_default_registries_populated():
    avail = registries.safety_registry.list_available()
    assert "joint_limit" in avail
    assert "torque_limit" in avail
    assert "self_collision" in avail
    assert "composite" in avail


def test_observer_registry_has_logger_recorder():
    avail = registries.observer_registry.list_available()
    assert "logger" in avail
    assert "recorder" in avail


def test_get_registry_dispatches_by_kind():
    assert registries.get_registry("controller.wbc") is registries.controller_registry
    assert registries.get_registry("safety.joint_limit") is registries.safety_registry
    with pytest.raises(ValueError):
        registries.get_registry("not_a_kind.foo")


# ---------------------------------------------------------------------------
# build_components — end-to-end on a synthetic preset
# ---------------------------------------------------------------------------


class _StubSpec:
    actuated_joints = ("j0", "j1")
    joint_range = {"j0": np.array([-1.0, 1.0]), "j1": np.array([-1.0, 1.0])}
    torque_limit = {"j0": 5.0, "j1": 5.0}


class _StubIO:
    def __init__(self) -> None:
        self.spec = _StubSpec()


def test_build_components_assembles_safety_with_spec():
    registries.io_registry.register("stub", _StubIO)

    class Preset(DeployConfig):
        class io(ComponentConfig):
            registry_key = "io.stub"

        class safety(ComponentConfig):
            registry_key = "safety.composite"
            filters = [
                {"registry_key": "safety.joint_limit"},
                {"registry_key": "safety.torque_limit"},
            ]

    built = build_components(Preset)
    assert isinstance(built, BuiltComponents)
    assert built.io is not None
    assert built.safety is not None
    assert len(built.safety.filters) == 2


def test_build_components_constructs_observers_list(tmp_path):
    class Preset(DeployConfig):
        observers = [
            {"registry_key": "observer.logger", "out_dir": str(tmp_path)},
        ]

    built = build_components(Preset)
    assert len(built.observers) == 1
    assert built.observers[0].name == "logger"


def test_build_components_skips_missing_sections():
    class Empty(DeployConfig):
        pass

    built = build_components(Empty)
    assert built.io is None
    assert built.controller is None
    assert built.safety is None
    assert built.observers == []
