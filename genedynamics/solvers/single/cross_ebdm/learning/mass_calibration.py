"""Bounded gain/delay/payload identification from sealed public fit probes.

The mass interval is a declared hypothesis, not a target construction value.
Held-out probes are deliberately absent from the fitting interface.
"""
from dataclasses import asdict, dataclass

import numpy as np

from genedynamics.tasks.transfer.runtime import MechanismSpec
from .physics_calibration import PublicPhysicsTwin, PublicProbe, probe_loss


@dataclass(frozen=True)
class PublicMassFamily:
    gain_min: float = .4
    gain_max: float = 1.3
    delays: tuple = (0, 1, 2, 3, 4)
    payload_min: float = .25
    payload_max: float = 1.25
    nominal_payload: float = .5
    gain_grid_points: int = 9
    mass_grid_points: int = 5
    free_refine_steps: int = 12
    joint_refine_delays: int = 5
    joint_max_evaluations: int = 90

    def __post_init__(self):
        values = (self.gain_min, self.gain_max, self.payload_min, self.payload_max, self.nominal_payload)
        if not np.isfinite(values).all() or not 0 < self.gain_min < self.gain_max:
            raise ValueError("invalid gain interval")
        if not 0 < self.payload_min <= self.nominal_payload <= self.payload_max or self.payload_min == self.payload_max:
            raise ValueError("invalid public payload interval")
        if not self.delays or any(isinstance(d, bool) or not isinstance(d, int) or d not in range(5) for d in self.delays):
            raise ValueError("invalid integer delay hypotheses")
        for name in ("gain_grid_points", "mass_grid_points", "joint_refine_delays", "joint_max_evaluations"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError("positive integer search ceilings required")
        if self.gain_grid_points < 2 or self.mass_grid_points < 2 or self.free_refine_steps < 0:
            raise ValueError("invalid grid/refinement ceilings")


def fit_public_mass_twin(spec, records, *, family=None, ledger=None, progress=None, cached_fit=None):
    """Profile every delay, then jointly refine a fixed number of fit winners.

    Free probes initialize gain at nominal mass; all fit probes select the
    payload profile and final bounded joint refinement. No validation-based
    retries, target truth, private carry or full-task outcomes are accepted.
    """
    from scipy.optimize import minimize, minimize_scalar

    family = family or PublicMassFamily()
    if not records or any(not isinstance(r, PublicProbe) or r.metadata["purpose"] != "fit" for r in records):
        raise ValueError("only sealed fit probes may select mechanism parameters")
    free = [r for r in records if r.metadata["kind"] == "free"]
    if not free or not any(r.metadata["kind"] == "loaded" for r in records):
        raise ValueError("free and loaded fit excitation required")
    calls, profile = [], []
    cache = {}
    inherited = 0
    if cached_fit is not None:
        from .physics_calibration import canonical
        if canonical(cached_fit["family"]) != canonical(asdict(family)) or cached_fit["fit_probe_sha256"] != [r.digest for r in records]:
            raise ValueError("cached model search family or fit probes changed")
        for entry in cached_fit["search"]:
            gain, delay, mass = entry["gain"], entry["delay"], entry["payload_mass"]
            if not (family.gain_min <= gain <= family.gain_max and delay in family.delays and
                    family.payload_min <= mass <= family.payload_max and np.isfinite(entry["loss"]) and entry["loss"] >= 0):
                raise ValueError("invalid cached model hypothesis")
            if entry["stage"] in {"free_grid", "free_refine"}:subset=free
            elif entry["stage"] in {"mass_profile", "joint_refine_fit_only"}:subset=records
            else:raise ValueError("cached search contains unknown or validation stage")
            key=(float(gain),int(delay),float(mass),tuple(r.digest for r in subset))
            if key not in cache:
                cache[key]=float(entry["loss"]);calls.append(dict(entry));inherited+=1

    def loss(gain, delay, mass, subset, stage):
        key = (float(gain), int(delay), float(mass), tuple(r.digest for r in subset))
        if key not in cache:
            twin = PublicPhysicsTwin(spec, MechanismSpec(float(gain), int(delay), float(mass)), ledger=ledger)
            value = float(np.mean([probe_loss(r.observations, twin.predict(r)[0], free=r.metadata["kind"] == "free") for r in subset]))
            cache[key] = value
            calls.append(dict(gain=float(gain), delay=int(delay), payload_mass=float(mass), loss=value, stage=stage))
            if progress is not None:
                progress(dict(search=calls, profile=profile))
        return cache[key]

    gains = np.unique(np.r_[np.linspace(family.gain_min, family.gain_max, family.gain_grid_points), 1.])
    masses = np.unique(np.r_[np.linspace(family.payload_min, family.payload_max, family.mass_grid_points), family.nominal_payload])
    for delay in family.delays:
        values = [loss(g, delay, family.nominal_payload, free, "free_grid") for g in gains]
        index = int(np.argmin(values)); gain = float(gains[index]); error = values[index]
        if family.free_refine_steps and error > 1e-14:
            refined = minimize_scalar(lambda g: loss(g, delay, family.nominal_payload, free, "free_refine"),
                method="bounded", bounds=(gains[max(0, index-1)], gains[min(len(gains)-1, index+1)]),
                options=dict(maxiter=family.free_refine_steps, xatol=1e-5))
            if refined.fun < error:
                gain = float(refined.x)
        mass_losses = [loss(gain, delay, mass, records, "mass_profile") for mass in masses]
        best_mass = int(np.argmin(mass_losses))
        profile.append(dict(gain=gain, delay=int(delay), payload_mass=float(masses[best_mass]), loss=mass_losses[best_mass],
            mass_profile=[dict(payload_mass=float(m), loss=v) for m, v in zip(masses, mass_losses)]))

    winners = sorted(profile, key=lambda r: (r["loss"], r["delay"]))[:family.joint_refine_delays]
    for row in winners:
        result = minimize(lambda x: loss(x[0], row["delay"], x[1], records, "joint_refine_fit_only"),
            [row["gain"], row["payload_mass"]], method="Powell",
            bounds=((family.gain_min, family.gain_max), (family.payload_min, family.payload_max)),
            options=dict(maxfev=family.joint_max_evaluations, xtol=1e-5, ftol=1e-8))
        # Retain every evaluated fit point, including an unfinished bounded
        # optimizer's best point, rather than its possibly worse final iterate.
        entries = [r for r in calls if r["delay"] == row["delay"] and r["stage"] in {"mass_profile", "joint_refine_fit_only"}]
        best = min(entries, key=lambda r: r["loss"])
        row.update(gain=best["gain"], payload_mass=best["payload_mass"], loss=best["loss"],
            optimizer_success=bool(result.success), optimizer_message=str(result.message))
    best = min(profile, key=lambda r: (r["loss"], r["delay"], r["gain"], r["payload_mass"]))
    mechanism = MechanismSpec(best["gain"], best["delay"], best["payload_mass"])
    return mechanism, dict(fit_loss=best["loss"], profile=profile, search=calls, family=asdict(family),
        fit_probe_sha256=[r.digest for r in records], validation_used_for_selection=False,
        unique_hypothesis_calls=len(calls),cached_hypothesis_calls=inherited,new_hypothesis_calls=len(calls)-inherited,
        scope="bounded public gain/integer-delay/payload family; cold resets only; no global or statistical identifiability certificate")
