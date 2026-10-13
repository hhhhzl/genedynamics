"""Observed-encoder reconstruction and finite actuator profile fitting."""
import numpy as np
from ...contracts import ExecutionSpec,MechanismSpec,RandomRequest,ControllerState
from ...cross_embodiment import CrossEmbodiment


class NumpyObstacleResponse:
    def __init__(self,config,skill,mechanism):
        self.runner=CrossEmbodiment(task=config,skill=skill,mechanism=mechanism,
            execution=ExecutionSpec(backend="numpy",physics_backend="mujoco"))

    def predict_probe(self,record):
        import mujoco
        initial=record.initial_state
        state=self.runner.reset_batch(RandomRequest(0,tuple(int(x) for x in initial.sample_id)))
        for data,observed in zip(state.physics,initial.physics):
            data.qpos[:]=observed.qpos;data.qvel[:]=observed.qvel
            mujoco.mj_forward(self.runner._impl.model,data)
        state=state._replace(controller=ControllerState(*(np.array(x,copy=True) for x in initial.controller)),
            steps=np.array(initial.steps,copy=True),hold_count=np.array(initial.hold_count,copy=True),status=np.array(initial.status,copy=True))
        return self.runner.collect_chunk(state,record.commands).trace


def features(trace):return np.concatenate((trace.tcp_position,trace.tcp_rotation.reshape(*trace.valid.shape,9)),axis=-1)


def fit_obstacle_context(config,skill,records):
    from scipy.optimize import minimize_scalar
    if not records or any(r.purpose!="fit" for r in records):raise ValueError("fit-purpose records only")
    scales=np.array([.0001]*3+[.001]*9);history=[]
    def objective(gain,delay):
        response=NumpyObstacleResponse(config,skill,MechanismSpec(float(gain),int(delay),config.nominal_payload_mass))
        error=np.concatenate([(((features(response.predict_probe(r))-features(r.result.trace))/scales)**2)[r.result.trace.valid] for r in records])
        value=float(error.mean());history.append([float(gain),int(delay),value]);return value
    grid=np.linspace(.4,1.3,9);coarse=[]
    for delay in range(5):
        scores=[objective(g,delay) for g in grid];i=int(np.argmin(scores));coarse.append((scores[i],float(grid[i]),delay,i))
    candidates=[]
    for value,gain,delay,index in sorted(coarse)[:2]:
        fit=minimize_scalar(lambda g:objective(g,delay),bounds=(grid[max(0,index-1)],grid[min(8,index+1)]),method="bounded",options={"xatol":1e-5,"maxiter":30})
        candidates.append((min(value,float(fit.fun)),float(fit.x) if fit.fun<value else gain,delay))
    error,gain,delay=min(candidates)
    return MechanismSpec(gain,delay,config.nominal_payload_mass),dict(fit_error=error,coarse_profiles=coarse,search=history,hypothesis_evaluations=len(history),
        scope="known nominal mass and morphology; finite delay profile with bounded gain fit; no hidden target model used")


def validate_obstacle_response(config,skill,mechanism,records):
    if not records or any(r.purpose!="validation" for r in records):raise ValueError("validation-purpose records only")
    response=NumpyObstacleResponse(config,skill,mechanism);errors=[];agreement=[]
    for r in records:
        prediction=response.predict_probe(r);errors.extend((features(prediction)-features(r.result.trace))[r.result.trace.valid].tolist())
        agreement.append(bool(np.array_equal(prediction.status,r.result.trace.status)))
    error=np.asarray(errors)
    position=float(np.sqrt(np.mean(error[:,:3]**2)));rotation=float(np.sqrt(np.mean(error[:,3:]**2)))
    return dict(position_rmse=position,rotation_matrix_rmse=rotation,status_agreement=all(agreement),
        thresholds=dict(position_rmse=.0005,rotation_matrix_rmse=.003),supported=position<=.0005 and rotation<=.003 and all(agreement))
