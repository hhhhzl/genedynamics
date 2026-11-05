# ASID solver

`ASIDSolver` exposes the finite-difference Fisher-information exploration
reward in the [authors' public release](https://github.com/WEIRDLabUW/asid).
The task supplies a simulator transition and chooses public actions.
The released repository describes its public code as an implementation of
this objective and includes task-specific exploration-policy training code. It
does not provide a reusable full system-identification and task-policy
retraining pipeline for the present four tasks. This package is therefore an
**exploration component**, not a
qualified four-task ASID baseline. Task observations, robot bindings, target
interaction accounting, and policy training belong outside this solver.

Task-owned development chains for T1--T4 execute Fisher-guided exploration,
same-action target-to-simulator parameter fitting, simulator-only task-policy
search, and held-out target deployment. They use independent CEM search in
this solver, including a predeclared coarse parameter grid before local
identification. This is **not** the authors' rod-specific SAC training.
The reports retain simulator and target interactions separately. Their audited
single training instances remain development evidence, not a qualified
four-task main-table baseline.

On disjoint T1 source, nominal, and gain-shift resets, the resulting policy
matches the public path exactly: 24/24 paired action streams and physical
traces are identical. This supports task-integration validity but provides no
independent T1 task-performance advantage for ASID.

The T3 development policy succeeds on 10/16 fresh paired requests, but its
actions are exactly identical to the public geometry actor on all 16. T4 has
an audited complete training chain; its fresh paired performance pilot is
recorded separately under the task-local `results/quadruped` tree. Neither
finding establishes a task-performance benefit from identification.

This is an independent implementation of the published mathematical objective.
The upstream repository has no license file, so no source code is copied here.
The pinned reference commit is `cf2cb670fb983f67cf2af156d4e17b4a690787e2`.
