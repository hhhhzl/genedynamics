"""Co-design-specific optimizer implementations (consume the JAX-MPM evaluator).
Registered into the co-design solver registry by solvers/single/codesign_solvers.py.
The GENERAL algorithms (CEM, and later CMA-ES) live in solvers/single/{cem,cmaes}
and do co-design via the (dynamics,energy) env bridge, not here."""
