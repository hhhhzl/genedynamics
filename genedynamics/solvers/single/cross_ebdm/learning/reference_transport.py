"""Portable, amortized affine transport fitted using a calibrated response.

The restricted action class provides a diagnostic between an oracle response
and a large residual MLP. The current fit optimizes normal-force scaling;
failure to beat simple inverse gain must remain a reported scientific result.
"""
import numpy as np
from .models import LearnedModel,ModelSpec
from .learned import FrozenRegressor
from .learning_data import Normalization


class ReferenceTransport:
    def __init__(self,regressor):
        if regressor.metadata.get("learned_role")!="reference_transport": raise ValueError("expected reference transport")
        self.regressor=regressor

    def apply(self,commands,phase=None):
        shape=commands.shape
        backend=self.regressor.model.backend
        if backend=="torch": commands=self.regressor.model._impl._tensor(commands)
        if self.regressor.metadata.get("phase_conditioned"):
            if phase is None: raise ValueError("phase-conditioned transport requires observed phase")
            if backend=="torch":
                import torch
                phase=torch.as_tensor(phase,device=commands.device)
                mask=(phase==3)[:,None,None].to(commands.dtype).expand(*shape[:2],1)
                if self.regressor.metadata.get("alignment_stage"):
                    align=(phase==2)[:,None,None].to(commands.dtype).expand(*shape[:2],1)
                    commands=torch.cat((commands,commands[...,12:13]*mask,align,mask),dim=-1)
                else: commands=torch.cat((commands,commands[...,12:13]*mask,mask),dim=-1)
            else:
                if backend=="jax":
                    import jax.numpy as xp
                else: xp=np
                commands=xp.asarray(commands,dtype=xp.float32)
                mask=xp.broadcast_to((xp.asarray(phase)==3)[:,None,None],(*shape[:2],1)).astype(commands.dtype)
                if self.regressor.metadata.get("alignment_stage"):
                    align=xp.broadcast_to((xp.asarray(phase)==2)[:,None,None],(*shape[:2],1)).astype(commands.dtype)
                    commands=xp.concatenate((commands,commands[...,12:13]*mask,align,mask),axis=-1)
                else: commands=xp.concatenate((commands,commands[...,12:13]*mask,mask),axis=-1)
        prediction=self.regressor.predict(commands.reshape(-1,commands.shape[-1]))[0]
        return prediction.reshape(shape)

    @classmethod
    def from_parameters(cls,scale,bias=(0.,0.),*,metadata=None,backend="jax",device="cpu"):
        """Observed phase gates final-insertion force and tangential correction.

        The two extra features are phase3*source_force and phase3. Expansion
        belongs to the input adapter; the portable model remains affine.
        """
        if np.shape(bias)!=(2,) or not np.isfinite(bias).all() or not .5<=scale<=2.5 or np.max(np.abs(bias))>.02: raise ValueError("outside declared transport support")
        model=LearnedModel(ModelSpec(15,13,hidden=()),backend=backend,device=device)
        weights=np.zeros((15,13),np.float32);weights[:13]=np.eye(13);weights[13,12]=scale-1.;weights[14,:2]=bias
        parameters=model.import_parameters({"layer.0.weight":weights[None],"layer.0.bias":np.zeros((1,13),np.float32)})
        norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
        return cls(FrozenRegressor(model,parameters,norm(15),norm(13),{
            **(metadata or {}),"learned_role":"reference_transport","force_scale":float(scale),"insertion_bias":list(map(float,bias)),
            "phase_conditioned":True,"support":"phase3 only: scale .5..2.5; tangential velocity bias +/- .02 m/s; common command bounds"}))

    @classmethod
    def from_stage_parameters(cls,scale,insertion_bias=(0.,0.),alignment_bias=(0.,0.),*,alignment_stiffness=(0.,)*6,insertion_stiffness=(0.,)*6,recovery_stiffness=(0.,)*6,metadata=None,backend="jax",device="cpu"):
        """Public phase gates; correction can act before tight-hole insertion."""
        if np.shape(insertion_bias)!=(2,) or np.shape(alignment_bias)!=(2,) or not np.isfinite([*insertion_bias,*alignment_bias]).all() or not .5<=scale<=2.5 or np.max(np.abs([*insertion_bias,*alignment_bias]))>.02:
            raise ValueError("outside declared stage transport support")
        if np.shape(alignment_stiffness)!=(6,) or np.shape(insertion_stiffness)!=(6,) or not np.isfinite([*alignment_stiffness,*insertion_stiffness]).all() or np.max(np.abs([*alignment_stiffness,*insertion_stiffness]))>1.:
            raise ValueError("stage stiffness outside shared log-stiffness support")
        if np.shape(recovery_stiffness)!=(6,) or not np.isfinite(recovery_stiffness).all() or np.max(np.abs(recovery_stiffness))>1.:
            raise ValueError("recovery stiffness outside shared log-stiffness support")
        model=LearnedModel(ModelSpec(16,13,hidden=()),backend=backend,device=device)
        weights=np.zeros((16,13),np.float32);weights[:13]=np.eye(13);weights[13,12]=scale-1.
        weights[14,:2]=alignment_bias;weights[15,:2]=insertion_bias
        # The constant offset acts in phases 0/1. Phase 2/3 replace it with
        # their independently declared offsets, rather than adding stiffness.
        weights[14,6:12]=np.asarray(alignment_stiffness)-recovery_stiffness
        weights[15,6:12]=np.asarray(insertion_stiffness)-recovery_stiffness
        bias=np.zeros((1,13),np.float32);bias[0,6:12]=recovery_stiffness
        parameters=model.import_parameters({"layer.0.weight":weights[None],"layer.0.bias":bias})
        norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
        return cls(FrozenRegressor(model,parameters,norm(16),norm(13),{
            **(metadata or {}),"learned_role":"reference_transport","force_scale":float(scale),
            "insertion_bias":list(map(float,insertion_bias)),"alignment_bias":list(map(float,alignment_bias)),
            "alignment_stiffness":list(map(float,alignment_stiffness)),"insertion_stiffness":list(map(float,insertion_stiffness)),
            "recovery_stiffness":list(map(float,recovery_stiffness)),
            "phase_conditioned":True,"alignment_stage":True,
            "support":"phase0/1, phase2 and phase3 independent log-stiffness +/-1; phase2/3 tangential bias +/- .02 m/s; phase3 scale .5..2.5; shared decoder bounds"}))

    @classmethod
    def from_scale(cls,scale,*,backend="jax",metadata=None):
        if not .5<=scale<=2.5: raise ValueError("force scale exceeds declared transport support")
        model=LearnedModel(ModelSpec(13,13,hidden=()),backend=backend)
        weights=np.eye(13,dtype=np.float32); weights[12,12]=scale
        tensors=((weights[None],np.zeros((1,13),np.float32)),)
        if backend=="jax":
            import jax.numpy as xp
            tensors=tuple(tuple(xp.asarray(x) for x in layer) for layer in tensors)
        elif backend=="torch": tensors=tuple(tuple(model._impl._tensor(x) for x in layer) for layer in tensors)
        norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
        regressor=FrozenRegressor(model,tensors,norm(13),norm(13),{
            **(metadata or {}),"learned_role":"reference_transport","force_scale":float(scale),
            "support":"source force 0..10 N, scale .5..2.5; controller bounds remain applied"})
        return cls(regressor)

    @classmethod
    def from_impedance(cls,force_scale,log_stiffness,*,backend="jax",device="cpu",metadata=None):
        """Finite all-phase compensation with unchanged spatial command law."""
        if not np.isfinite(force_scale) or not .5<=force_scale<=2.5 or np.shape(log_stiffness)!=(6,) or not np.isfinite(log_stiffness).all() or np.max(np.abs(log_stiffness))>1.:
            raise ValueError("outside shared impedance compensation support")
        model=LearnedModel(ModelSpec(13,13,hidden=()),backend=backend,device=device)
        weights=np.eye(13,dtype=np.float32);weights[12,12]=force_scale
        bias=np.zeros(13,np.float32);bias[6:12]=log_stiffness
        parameters=model.import_parameters({"layer.0.weight":weights[None],"layer.0.bias":bias[None]})
        norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
        return cls(FrozenRegressor(model,parameters,norm(13),norm(13),dict(metadata or {},
            learned_role="reference_transport",force_scale=force_scale,log_stiffness=list(log_stiffness),
            support="all-phase finite impedance compensation; common decoder bounds and force limits")))

    def save(self,path):self.regressor.save(path)

    @classmethod
    def load(cls,path,*,backend="jax",device="cpu"):
        return cls(FrozenRegressor.load(path,backend=backend,device=device))
