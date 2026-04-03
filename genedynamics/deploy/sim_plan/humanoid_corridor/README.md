# Humanoid Corridor Follower Skeleton

This package is the file-level scaffold for a G1 corridor follower:

- `schema.py`
  Defines the typed plan/task objects shared by every phase.
- `trajectory_adapter.py`
  Owns `trajectory.json -> typed plan` loading and 14D/16D schema adaptation.
- `phase_scheduler.py`
  Stage-1 quasi-static support/swing scheduler.
- `task_generator.py`
  Turns plan frames into pelvis, foot, torso, and arm tasks.
- `g1_model.py`
  Collects G1 joint/site names and MuJoCo indices.
- `solver_base.py`
  Stable solver interface that both IK and WBC can implement.
- `ik_solver.py`
  Stage-1 MuJoCo IK backend with foot-position IK and joint-hint mapping.
- `joint_tracker.py`
  Stage-2-ready smoothing and safety clamps for joint references.
- `wbc_solver.py`
  Phase-3 placeholder for a future WBC/QP backend.
- `follower.py`
  End-to-end orchestrator for offline MuJoCo rollouts.

Extension path:

1. Phase 1
   Flesh out the task generator heuristics and improve the IK seed / foot IK.
2. Phase 2
   Add balance feedback, contact gating, touchdown handling, and stronger safety
   logic around `phase_scheduler.py` and `joint_tracker.py`.
3. Phase 3
   Swap `G1CorridorIKSolver` for a WBC implementation behind the unchanged
   `CorridorTaskSolverBase.solve()` interface.
