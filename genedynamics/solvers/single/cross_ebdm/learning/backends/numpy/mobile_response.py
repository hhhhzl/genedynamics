"""Solver-style finite coupled-response qualification for a known mobile body."""
import numpy as np
from ...structured_response import PhysicsTwinResponse,response_features


def mobile_features(trace):
    return np.concatenate((response_features(trace),trace.base_position,trace.base_tilt[...,None],trace.support_count[...,None]),axis=-1)


def fit_mobile_context(config,records,family,*,backend="numpy",device="cpu"):
    if backend!="numpy" or device!="cpu":raise NotImplementedError("mobile twin search currently numpy/cpu only")
    if not records or any(r.purpose!="fit" for r in records):raise ValueError("fit records only")
    if not family:raise ValueError("declared finite mobile family required")
    profile=[];scale=np.array([.001]*3+[.03]*9+[1.,2.]+[.002]*3+[.01,.1])
    for mechanism in family:
        twin=PhysicsTwinResponse(config,mechanism);errors=[]
        for record in records:
            difference=mobile_features(twin.predict_probe(record))-mobile_features(record.result.trace)
            errors.extend(np.mean((difference/scale)**2,axis=-1)[record.result.trace.valid].tolist())
        profile.append(dict(mechanism=mechanism.__dict__,error=float(np.mean(errors))))
    index=min(range(len(family)),key=lambda i:profile[i]["error"])
    return family[index],dict(profile=profile,model_probe_replays=len(family)*len(records),
        scope="finite coupled family, observed encoders and known full controller carry; no global identifiability certificate")


def validate_mobile_response(config,mechanism,records):
    if not records or any(r.purpose!="validation" for r in records):raise ValueError("independent validation probes required")
    twin=PhysicsTwinResponse(config,mechanism);errors=[]
    for record in records:
        errors.append((mobile_features(twin.predict_probe(record))-mobile_features(record.result.trace))[record.result.trace.valid])
    e=np.concatenate(errors)
    values=dict(position_rmse_m=float(np.sqrt(np.mean(e[:,:3]**2))),rotation_matrix_rmse=float(np.sqrt(np.mean(e[:,3:12]**2))),
        force_rmse_N=float(np.sqrt(np.mean(e[:,12]**2))),force_peak_rmse_N=float(np.sqrt(np.mean(e[:,13]**2))),
        body_position_rmse_m=float(np.sqrt(np.mean(e[:,14:17]**2))),body_tilt_rmse_rad=float(np.sqrt(np.mean(e[:,17]**2))),
        support_agreement=float(np.mean(e[:,18]==0)))
    limits=dict(position_rmse_m=.001,rotation_matrix_rmse=.03,force_rmse_N=.25,force_peak_rmse_N=1.,body_position_rmse_m=.002,body_tilt_rmse_rad=.01)
    return dict(**values,thresholds=limits,supported=all(values[k]<v for k,v in limits.items()) and values["support_agreement"]==1.,control_steps=len(e))
