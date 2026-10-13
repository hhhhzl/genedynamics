"""CPU physics-twin response provider and constrained mechanism identification.

The public hypothesis family specifies known nominal mass and unknown gain /
integer command delay. This deliberately does not infer unobservable mass or
claim a learned world model. Provider calls can later be replaced by a batched
JAX/native implementation without changing fit/tune/evaluation orchestration.
"""
from dataclasses import dataclass
import numpy as np


def response_features(trace):
    return np.concatenate((trace.tcp_position,trace.tcp_rotation.reshape(*trace.tcp_position.shape[:-1],9),
                           trace.normal_force[...,None],trace.force_peak[...,None]),axis=-1)


@dataclass(frozen=True)
class MechanismFamily:
    gain_min: float=.4
    gain_max: float=1.3
    delays: tuple=(0,1,2,3,4)
    known_payload_mass: float=.5

    def __post_init__(self):
        if not 0<self.gain_min<self.gain_max or not self.delays or any(d not in range(5) for d in self.delays):
            raise ValueError("invalid physical hypothesis family")


class PhysicsTwinResponse:
    """Lazy solver-style frontend; provider owns physics state and fitting."""
    def __init__(self,config,mechanism,*,backend="numpy",device="cpu"):
        if backend!="numpy" or device!="cpu":
            raise NotImplementedError("structured twin currently qualifies only numpy/cpu; no implicit fallback")
        from .backends.numpy.structured_response import NumpyPhysicsTwinResponse
        self._impl=NumpyPhysicsTwinResponse(config,mechanism)
        self.backend,self.device=backend,device

    @property
    def capabilities(self):return self._impl.capabilities

    def predict_probe(self,record):return self._impl.predict_probe(record)


def fit_twin_context(config,records,*,family=None,grid_points=9,refine=True,backend="numpy",device="cpu"):
    if backend!="numpy" or device!="cpu":
        raise NotImplementedError("structured identification currently qualifies only numpy/cpu")
    from .backends.numpy.structured_response import fit_twin_context as fit
    return fit(config,records,family=family,grid_points=grid_points,refine=refine)


def fit_contact_twin(config,records,*,frictions=(.15,.4,.7),time_constants=(.01,.02,.04),backend="numpy",device="cpu"):
    """Gain/delay profile followed by a declared finite contact family."""
    if backend!="numpy" or device!="cpu": raise NotImplementedError("contact hypothesis search currently numpy/cpu only")
    from .backends.numpy.structured_response import fit_contact_twin as fit
    return fit(config,records,frictions=frictions,time_constants=time_constants)


def validate_twin(config,mechanism,records):
    model=PhysicsTwinResponse(config,mechanism); errors=[]
    for record in records:
        p=response_features(model.predict_probe(record)); y=response_features(record.result.trace)
        errors.append((p-y)[record.result.trace.valid])
    if not errors: raise ValueError("independent validation records required")
    errors=np.concatenate(errors)
    report={"position_rmse_m":float(np.sqrt(np.mean(errors[:,:3]**2))),
            "rotation_matrix_rmse":float(np.sqrt(np.mean(errors[:,3:12]**2))),
            "force_rmse_N":float(np.sqrt(np.mean(errors[:,12]**2))),
            "force_peak_rmse_N":float(np.sqrt(np.mean(errors[:,13]**2))),"control_steps":len(errors)}
    report["supported"]=bool(report["position_rmse_m"]<.001 and report["force_rmse_N"]<.25 and report["force_peak_rmse_N"]<1.)
    report["thresholds"]={"position_rmse_m":.001,"force_rmse_N":.25,"force_peak_rmse_N":1.}
    return report
