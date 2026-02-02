import os
import pickle
import glob
import sys
from pathlib import Path
import torch
import importlib
from collections import namedtuple

# DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')
DiffusionExperiment = namedtuple("Diffusion", "dataset model diffusion trainer epoch losses")


def _ensure_diffuser_on_path():
    """Guarantee that the upstream DPCC `diffuser` package is importable.

    When using the vendored DPCC implementation, the trained checkpoints may
    still reference modules from an external checkout (e.g., ../dpcc/diffuser).
    This helper mirrors the search order used in the MethodPlugin so that
    unpickling works even in subprocesses where sys.path was not yet patched.
    """

    project_root = Path(__file__).resolve().parents[6]  # repo root (/.../enerdynamics)
    # NOTE: iterate low→high priority but insert with sys.path.insert(0),
    # so later entries end up ahead in import resolution.
    candidates = [
        project_root / "dpcc",         # child in repo (lowest)
        project_root.parent / "dpcc",  # sibling repo
    ]

    for cand in candidates:
        cand = cand.resolve()
        if not cand.exists():
            continue
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        diffuser_subdir = cand / "diffuser"
        if diffuser_subdir.exists() and str(diffuser_subdir) not in sys.path:
            sys.path.insert(0, str(diffuser_subdir))


def _alias_patch_diffusion():
    """Route `diffuser.models.diffusion` imports to the patched local implementation.

    Checkpoints may reference the upstream module path, but we want the patched
    `GaussianDiffusion` in `enerdynamics/solvers/single/dpcc/patch/diffusion.py`.
    """

    from enerdynamics.solvers.single.dpcc.patch import diffusion as patch_diffusion

    # Import the real diffuser package (from sys.path set above) so that utils/*
    # and other helpers remain available for unpickling. Then override the diffusion
    # module reference with the patched implementation.
    try:
        diffuser_pkg = importlib.import_module("diffuser")
        models_pkg = importlib.import_module("diffuser.models")
    except ModuleNotFoundError:
        # If the package truly is absent, fall back to a lightweight shim.
        import types
        diffuser_pkg = sys.modules.setdefault("diffuser", types.ModuleType("diffuser"))
        diffuser_pkg.__path__ = []
        models_pkg = sys.modules.setdefault("diffuser.models", types.ModuleType("diffuser.models"))

    sys.modules["diffuser.models.diffusion"] = patch_diffusion
    models_pkg.diffusion = patch_diffusion
    diffuser_pkg.models = models_pkg


# Ensure availability at import time (covers fork/spawn workers).
_ensure_diffuser_on_path()
_alias_patch_diffusion()


def mkdir(savepath):
    """
    returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False


def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), "state_*")
    latest_epoch = -1
    for state in states:
        try:
            epoch = int(state.replace("state_", "").replace(".pt", ""))
        except ValueError:
            epoch = -1
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch


def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, "rb"))
    return config


def load_losses(*loadpath):
    loadpath = os.path.join(*loadpath)
    if os.path.exists(loadpath):
        losses = pickle.load(open(loadpath, "rb"))
        return losses
    else:
        return None


def load_diffusion(*loadpath, epoch="latest", device="cuda:0", seed=None):
    print(f"\n[ utils/serialization ] Loading model from {os.path.join(*loadpath)}\n")
    dataset_config = load_config(*loadpath, "dataset_config.pkl")
    model_config = load_config(*loadpath, "model_config.pkl")
    diffusion_config = load_config(*loadpath, "diffusion_config.pkl")
    trainer_config = load_config(*loadpath, "trainer_config.pkl")

    trainer_config._dict["results_folder"] = os.path.join(*loadpath)

    dataset = dataset_config()
    model = model_config().to(device)
    diffusion = diffusion_config(model).to(device)
    trainer = trainer_config(diffusion_model=diffusion, dataset=dataset)

    if epoch == "latest":
        epoch = get_latest_epoch(loadpath)

    # The upstream trainer.load is strict on state_dict keys; patch to allow
    # extra keys (e.g., loss_fn.weights) emitted by newer checkpoints.
    def _relaxed_load(ep):
        loadpath_ = os.path.join(trainer.logdir, f"state_{ep}.pt")
        data = torch.load(loadpath_, map_location=device)
        trainer.step = data["step"]
        trainer.model.load_state_dict(data["model"], strict=False)
        trainer.ema_model.load_state_dict(data["ema"], strict=False)

    trainer.load = _relaxed_load
    trainer.load(epoch)
    losses = load_losses(*loadpath, "losses.pkl")

    return DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer, epoch, losses)


def check_compatibility(experiment_1, experiment_2):
    """
    returns True if `experiment_1 and `experiment_2` have the same normalizers and timesteps
    """
    normalizers_1 = experiment_1.dataset.normalizer.get_field_normalizers()
    normalizers_2 = experiment_2.dataset.normalizer.get_field_normalizers()
    for key in normalizers_1:
        norm_1 = type(normalizers_1[key])
        norm_2 = type(normalizers_2[key])
        assert norm_1 == norm_2, f"Normalizers should match, found {norm_1} vs {norm_2} for field {key}"

    n_steps_1 = experiment_1.diffusion.n_timesteps
    n_steps_2 = experiment_2.diffusion.n_timesteps
    assert n_steps_1 == n_steps_2, (
        "Number of timesteps should match between diffusion experiments, "
        f"found {n_steps_1} and {n_steps_2}"
    )
