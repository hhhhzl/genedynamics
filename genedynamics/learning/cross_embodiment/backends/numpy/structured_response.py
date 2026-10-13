"""MuJoCo twin and SciPy identification implementation for the NumPy provider."""
import numpy as np
from ...contracts import MechanismSpec,RandomRequest
from ...contact import ContactRunner
from ...structured_response import MechanismFamily,response_features


class NumpyPhysicsTwinResponse:
    backend="numpy"
    capabilities={"physical_response","integer_delay","full_controller_carry","cpu"}

    def __init__(self,config,mechanism):
        self.runner=ContactRunner(config=config,mechanism=mechanism)

    def predict_probe(self,record):
        import mujoco
        request=RandomRequest(0,tuple(int(v) for v in record.initial_state.sample_id))
        state=self.runner.reset_batch(request)
        # Read observed initial coordinates; never copy a hidden target model.
        data=state.physics[0]; measured=record.initial_state.physics[0]
        data.qpos[:]=measured.qpos; data.qvel[:]=measured.qvel
        data.qacc_warmstart[:]=0.
        mujoco.mj_forward(self.runner._impl.model,data)
        fields={name:getattr(record.initial_state,name).copy() for name in state._fields if name!="physics"}
        state=state._replace(**fields)
        return self.runner.collect_chunk(state,record.commands).trace


def fit_twin_context(config,records,*,family=None,grid_points=9,refine=True):
    """Constrained profile likelihood: delay discrete, gain bounded continuous.

    Only fit-purpose records may enter. Validation never selects parameters.
    All simulator hypotheses and fitting calls are reported separately from the
    counted target interactions. A fit boundary is a warning, not confidence.
    """
    from scipy.optimize import minimize_scalar
    family=family or MechanismFamily()
    if not records or any(record.purpose!="fit" for record in records): raise ValueError("fit records only")
    observed=[response_features(r.result.trace) for r in records]
    scales=np.array([.002]*3+[.03]*9+[1.,2.])
    calls=[]
    free=[i for i,r in enumerate(records) if np.max(r.commands[...,12])<=.01]
    if not free: raise ValueError("excited free-space fit record required for gain/delay separation")
    def error(gain,delay,*,free_only=False):
        model=NumpyPhysicsTwinResponse(config,MechanismSpec(float(gain),int(delay),family.known_payload_mass))
        errors=[]
        indices=free if free_only else range(len(records))
        for i in indices:
            record,y=records[i],observed[i]
            prediction=response_features(model.predict_probe(record))
            mask=record.result.trace.valid
            local_scales=np.array([.0001]*3+[.001]*9+[1.,2.]) if i in free else scales
            errors.extend(np.mean(((prediction-y)/local_scales)**2,axis=-1)[mask].tolist())
        value=float(np.mean(errors)); calls.append([float(gain),int(delay),value,"free" if free_only else "joint"]); return value
    candidates=[]
    grid=np.linspace(family.gain_min,family.gain_max,grid_points)
    for delay in family.delays:
        values=[error(g,delay,free_only=True) for g in grid]; best=int(np.argmin(values))
        gain=grid[best]
        if refine:
            left=grid[max(0,best-1)]; right=grid[min(len(grid)-1,best+1)]
            result=minimize_scalar(lambda g:error(g,delay,free_only=True),bounds=(left,right),method="bounded",options={"xatol":1e-5,"maxiter":30})
            if result.fun<values[best]: gain=float(result.x)
        # Free-space excitation separates actuator gain/delay before the
        # discontinuous contact cost. Joint fitting stays local to that fit.
        value=error(gain,delay)
        if refine:
            left=max(family.gain_min,gain-.002);right=min(family.gain_max,gain+.002)
            result=minimize_scalar(lambda g:error(g,delay),bounds=(left,right),method="bounded",options={"xatol":1e-5,"maxiter":20})
            if result.fun<value:gain,value=float(result.x),float(result.fun)
        candidates.append((value,gain,delay))
        print({"delay_hypothesis":delay,"gain":float(gain),"fit_error":float(value)},flush=True)
    candidates.sort()
    best,gain,delay=candidates[0]
    return MechanismSpec(gain,delay,family.known_payload_mass),{
        "fit_error":best,"profile":[list(v) for v in candidates],"hypothesis_evaluations":len(calls),"search":calls,
        "protocol":"excited free-space profile then local joint fit; no global certificate",
        "family":family.__dict__,"boundary":bool(min(gain-family.gain_min,family.gain_max-gain)<.005),
        "scope":"structured twin, known nominal mass; no general contact parameter identifiability claim"}


def fit_contact_twin(config,records,*,frictions,time_constants):
    from ...contact import ContactMechanismSpec
    if not frictions or not time_constants or any(v<=0 or not np.isfinite(v) for v in (*frictions,*time_constants)):
        raise ValueError("finite positive contact family required")
    free=[r for r in records if np.max(r.commands[...,12])<=.01]
    actuator,actuator_report=fit_twin_context(config,free)
    contact=[r for r in records if np.max(r.commands[...,12])>.01]
    if not contact or any(r.purpose!="fit" for r in records):raise ValueError("fit contact records required")
    candidates=[];signatures=[];scales=np.array([.002]*3+[.03]*9+[1.,2.])
    for friction in frictions:
        for time_constant in time_constants:
            mechanism=ContactMechanismSpec(actuator.servo_gain,actuator.delay_steps,actuator.payload_mass,float(friction),float(time_constant))
            twin=NumpyPhysicsTwinResponse(config,mechanism);errors=[];signature=[]
            for record in contact:
                prediction=response_features(twin.predict_probe(record));observed=response_features(record.result.trace)
                valid=record.result.trace.valid
                errors.extend(np.mean(((prediction-observed)/scales)**2,axis=-1)[valid].tolist())
                signature.extend((prediction/scales)[valid].ravel().tolist())
            candidates.append((float(np.mean(errors)),mechanism));signatures.append(signature)
    best,mechanism=min(candidates,key=lambda row:row[0])
    from ...research_audit import ResearchAudit
    separation=ResearchAudit().separation(signatures,noise_bound=0.,model_bias_bound=0.)
    return mechanism,{"actuator_fit":actuator_report,"fit_error":best,
        "contact_profile":[dict(error=e,**m.__dict__) for e,m in candidates],
        "contact_signature_diagnostic":separation,"hypothesis_evaluations":len(candidates),
        "scope":"finite contact profile conditional on fitted actuator; zero-error signature diagnostic is not a noisy/global certificate"}

