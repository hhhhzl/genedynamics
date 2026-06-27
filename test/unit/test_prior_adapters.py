"""Stage 6 — modern-prior adapters (Table 3) registration + lazy behavior.

CPU, no weights: importing the priors package registers all Table-3 adapters;
constructing an uninstalled one raises MissingDependencyError with an install
hint (the lazy contract). No GPU / no 3D-gen inference here.
"""

from __future__ import annotations

import pytest

from genedynamics.learning.priors.morphology.generators import (
    list_priors,
    get_prior,
    MissingDependencyError,
)

MODERN = ["triposg", "hunyuan3d", "trellis", "craftsman", "meshflow", "diffgs"]


def test_all_modern_priors_registered():
    names = list_priors()
    assert "random_shapes" in names
    for n in MODERN:
        assert n in names, f"{n} not registered; got {names}"


@pytest.mark.parametrize("name", MODERN)
def test_uninstalled_prior_raises_missing_dependency(name):
    with pytest.raises(MissingDependencyError) as ei:
        get_prior(name)
    # The error carries a non-empty install hint for the user.
    assert getattr(ei.value, "install_hint", "")


def test_random_shapes_prior_actually_samples():
    p = get_prior("random_shapes")
    meshes = p.sample("a worm-like crawler", n=2, seed=0)
    assert len(meshes) == 2
