# Cross-EBDM learning

This package owns learned components used only by Cross-EBDM: calibration,
response models, transport models, checkpoint payloads, and backend-specific
training/inference kernels. It does not own shared datasets, task physics,
experiment result paths, or paper workflows.

The ownership boundary is:

- `genedynamics.tasks.transfer` defines the four task contracts and public observations.
- `genedynamics.envs.domains.manipulation.transfer_workcell` owns simulation.
- `genedynamics.solvers.single.cross_ebdm.learning` fits and loads learned models.
- `genedynamics.solvers.single.cross_ebdm` owns Cross-EBDM inference, transport, and planning logic.
- `genedynamics.experiments.plugins.methods.transfer` binds methods to the four-task runner.
- `scripts/paper/cross_ebdm` contains the paper-facing matrix commands.

Training code must write checkpoints with the dataset identity, split, feature
schema, normalization, seed, model configuration, and source revision. Solver
code consumes those frozen artifacts and does not update them during evaluation.
Optional native inference is built explicitly with:

```bash
python -m genedynamics.solvers.single.cross_ebdm.learning.backends.native.build \
  --output .runtime/cross_ebdm/native/libcross_ebdm.so
```

Set `CROSS_EBDM_NATIVE_LIBRARY` to that library only for the optional CPU
native backend. Torch, JAX, and NumPy backends remain available through their
package modules.
