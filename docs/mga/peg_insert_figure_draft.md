# PegInsert figure draft

This is a visual-design and data-availability check, not a new experimental
results section. No solver, experiment configuration, or recorded result is
modified by the plotting script.

## Reproduce

Run `scripts/paper/mga/plot_peg_insert_draft.py` with Python, NumPy, and
Matplotlib. It reads the recorded `results.json` and `trajectory.json` files
under `results/arm/peg_insert/`, and writes vector PDFs, PNG previews, and a
source/summary JSON under `latex/latex_mga/output/pdf/`.

The three suites are `id_wide`, `ood_pose`, and `ood_sensing`; every panel
includes seeds 0-9. The records use the guided structured-bank configuration
(56 Gaussian and 8 stochastic RL candidates for MGA), even though the current
MGA YAML has since changed to additive proposal selection. The recorded config
snapshots, not the current YAML, determine this draft's interpretation.

## Execution-space trajectory style

`mga-peg-insert-trajectories.pdf` compares Standalone RL, MGA without the RL
prior, and MGA. The middle column is the actual `ablation/no_rl_prior` run,
not the separate `baseline/model_based_only` controller, which also differs in
other mechanisms.

- Coordinates: recorded lateral error and insertion depth, in millimeters.
- Thin gray paths: complete executed rollouts; red segments indicate a recorded
  force/torque violation at the segment's destination.
- Hollow markers: final recorded states.
- Blue target region: insertion depth at least 32 mm and lateral error at most
  1.2 mm, the default positional success tolerances for this task. This is not
  the complete feasible set: orientation, contact limits, and success hold time
  also matter. A point inside the blue region need not be safe or successful.
- Safe-success counts come directly from the saved run metrics.
- The full fixed-length records are retained, including states after the first
  success flag, to match the reported full-run safety metrics. Deterministic RL
  rollouts can overlap exactly across seeds; overlapping lines are not removed.

These paths are **executed trajectories, not sampled candidate horizons**.
They demonstrate a potential visual style and do not establish a sampling
advantage for the learned prior.

## Source-level proposal diagnostic

`mga-peg-insert-proposal-weights.pdf` uses MGA's saved `infos` fields
`proposal_rl_weight` and `proposal_gaussian_weight`. At each replanning step it
plots `W_RL / (W_RL + W_G)`, excluding deterministic anchors. Thin lines show
all recorded runs; the teal line is their median. The dashed `8/64` reference
corresponds to equal average weight per stochastic candidate.

The saved source weights are already averaged over the reverse steps of each
replan. The plotted quantity is a ratio of those averaged source weights, not
the average of per-reverse-step ratios. It measures model-weight contribution
within the existing mixed bank; it is not a safety probability, a selection
frequency, or a controlled comparison of equal-budget standalone samplers.
Gaussian slots in this guided bank already inherit the active RL warm start
and trust map, so they are not an uninformed Gaussian baseline.

## What is still needed for a genuine proposal-trajectory figure

The saved files do not contain individual candidate node sequences or their
rollout states. A same-state candidate-path figure needs a separate small
diagnostic that records, before weighting/refinement:

1. A shared starting state, plan center, reverse index, and noise schedule.
2. Source-labeled candidate nodes after the common coordinate/action maps.
3. Re-rolled physical states and force/torque traces for each candidate.
4. Candidate counts, all active maps, and model-versus-execution parameters.

An RL-centered Gaussian control can distinguish a better proposal center from
additional state-conditioned trajectory structure. The current figures must
not be relabeled as that controlled experiment.

## Algorithm scope

The algorithm added to `latex/latex_mga/tex/algorithm.tex` summarizes the
structured-bank formulation currently written in the paper and used by these
saved experiments. It does not describe the newer additive expert-selection
variant. Updating that narrative is a separate authorial decision.
