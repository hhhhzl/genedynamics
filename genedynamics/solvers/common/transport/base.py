"""
Reverse-transport interface for the MBD-family solvers.

The reverse step of every model-based-diffusion solver is the innermost
recombination

    tau_tilde_{k-1} = T_k^{rho_k}( tau_k, tau1_k, eps_k, score_g, sched, sigma, z )

where (see docs/methods/transport_unification_plan.md and
latex_2go/sections/method.tex, eq:unified_reverse_update_k):

  - ``tau_k``      : current noisy iterate ``Yi`` ( = Ybar * sqrt(abar_k) )
  - ``tau1_k``     : reward-weighted clean estimate ``Ybar_weighted`` (tau-hat_{1|k})
  - ``eps_k``      : residual / predicted noise direction (eps-hat_k); not used by DDPM
  - ``score_g``    : optional geometry-drift score term (None for plain MBD)
  - ``sched``      : per-step scheduler scalars (dict): ``abar_k``, ``abar_km1``,
                     ``alpha_k`` (and family-specific scalars added later).
  - ``sigma``      : optional reverse-noise scale ``sigma_k`` (default 0 =
                     deterministic). DDIM vs FM differ only when ``sigma > 0``.
  - ``z``          : optional unit-variance noise ``z_k`` for the ``sigma * z`` kick.

A transport's ``step`` returns the reverse iterate
``tau_tilde_{k-1}`` (the renormalised ``Ybar_next``). With the default
``sigma = 0`` the step is deterministic. The solver-specific
stochastic terms (the annealed ``extra_sigma`` diversity kick, geometry tangent
noise, CFS retraction, ...) compose on top, unchanged, in the solver body.

**None-default contract.** When a solver's ``transport`` attribute is ``None``
the solver runs its *verbatim* inline DDPM lines, so the traced graph (and
therefore the float output) is byte-identical to the pre-transport code. A
transport object is only consulted on the explicit ``else`` branch. The
``DDPMTransport`` backend reproduces the inline default exactly so that
``transport=DDPMTransport()`` matches the ``transport=None`` baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional


class ReverseTransport(ABC):
    """Abstract reverse-transport operator ``T_k^{rho}``."""

    #: Human-readable family tag ("DDPM" | "DDIM" | "FM" | ...).
    family: str = "ABSTRACT"

    #: Whether ``step`` itself consumes the reverse-noise ``sigma * z`` kick.
    #: DDPM is ``False`` (it ignores ``sigma``/``z``; the MBD solver adds its
    #: own annealed ``extra_sigma`` diversity term *after* ``step``). DDIM/FM are
    #: ``True`` (they apply the ``sigma * z`` kick inside ``step``), so the solver
    #: must hand its per-step diversity ``sigma``/``z`` to ``step`` and NOT
    #: re-add the diversity term afterward (which would double-count the noise).
    consumes_diversity_noise: bool = False

    @abstractmethod
    def step(
        self,
        tau_k: Any,
        tau1_k: Any,
        eps_k: Optional[Any],
        score_g: Optional[Any],
        sched: Mapping[str, Any],
        sigma: Any = 0.0,
        z: Optional[Any] = None,
    ) -> Any:
        """Return the reverse iterate ``tau_tilde_{k-1}``.

        Args:
            tau_k:   current noisy iterate (``Yi``).
            tau1_k:  reward-weighted clean estimate (``Ybar_weighted``).
            eps_k:   residual / predicted-noise direction (may be ``None``;
                     unused by DDPM, needed by DDIM/FM).
            score_g: optional geometry-drift score term (``None`` for plain MBD).
            sched:   per-step scheduler scalars (``abar_k``, ``abar_km1``,
                     ``alpha_k``, ...).
            sigma:   optional reverse-noise scale ``sigma_k`` (default ``0`` =
                     deterministic). DDIM and FM differ only when ``sigma > 0``
                     (see latex_2go/sections/prelims.tex, eq:ddim_update_2go and
                     eq:fm_update_expanded_2go).
            z:       optional unit-variance noise ``z_k``. The ``sigma * z`` kick
                     is added only when both ``sigma`` and ``z`` are supplied.
                     The MBD solver leaves these at their defaults and applies
                     its own annealed ``extra_sigma`` diversity term on top.

        Returns:
            ``tau_tilde_{k-1}`` (renormalised ``Ybar_next``), before any
            solver-specific stochastic / geometric post-terms.
        """
        raise NotImplementedError
