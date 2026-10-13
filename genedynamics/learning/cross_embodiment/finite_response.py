"""Finite empirical joint response laws with portable explicit-noise sampling.

The response vector is sampled jointly, preserving correlations and atoms at
failure events. This is a finite-policy distribution estimator, not a neural
autoregressive dynamics model or a state-conditional universal kernel.
"""
import numpy as np
from .learned import FrozenRegressor
from .models import LearnedModel,ModelSpec
from .learning_data import Normalization


class FiniteResponseLaw:
    def __init__(self,regressor):
        if regressor.metadata.get("learned_role")!="finite_joint_response_law":raise ValueError("finite joint response law required")
        self.regressor=regressor
        self.cases=int(regressor.metadata["cases"]);self.samples=int(regressor.metadata["samples_per_case"])
        if regressor.model.spec.input_dim!=self.cases*self.samples:raise ValueError("response support layout mismatch")

    @classmethod
    def fit(cls,table,*,backend="jax",device="cpu",metadata=None):
        table=np.asarray(table,np.float32)
        if table.ndim!=3 or min(table.shape)<1 or not np.isfinite(table).all():raise ValueError("finite [case,sample,response] data required")
        c,n,d=table.shape;model=LearnedModel(ModelSpec(c*n,d,hidden=()),backend=backend,device=device)
        parameters=model.import_parameters({"layer.0.weight":table.reshape(1,c*n,d),"layer.0.bias":np.zeros((1,d),np.float32)})
        norm=lambda k:Normalization(np.zeros(k,np.float32),np.ones(k,np.float32))
        return cls(FrozenRegressor(model,parameters,norm(c*n),norm(d),dict(metadata or {},
            learned_role="finite_joint_response_law",cases=c,samples_per_case=n,
            estimator="empirical conditional law, uniform weights, joint response vectors",
            scope="declared finite cases and calibrated context; reset distribution integrated out; no arbitrary-action or initial-state extrapolation")))

    def sample(self,cases,uniform):
        if np.shape(cases)!=np.shape(uniform) or len(np.shape(cases))!=1:raise ValueError("case and uniform must be [batch]")
        backend=self.regressor.model.backend
        traced=False
        if backend=="jax":
            import jax
            traced=any(isinstance(x,jax.core.Tracer) for x in (cases,uniform))
        if backend!="torch" and not traced:
            ids=np.asarray(cases);values=np.asarray(uniform)
            if (not np.isfinite(ids).all() or not np.isfinite(values).all() or
                np.any(ids!=np.floor(ids)) or np.any(ids<0) or np.any(ids>=self.cases) or np.any(values<0) or np.any(values>=1)):
                raise ValueError("outside finite response support")
        if backend=="torch":
            import torch
            cases=torch.as_tensor(cases,device=self.regressor.model._impl.device)
            u=torch.as_tensor(uniform,device=cases.device)
            if bool((~torch.isfinite(cases)|~torch.isfinite(u)|(cases!=cases.floor())|(cases<0)|(cases>=self.cases)|(u<0)|(u>=1)).any()):raise ValueError("outside finite response support")
            cases=cases.to(torch.int64)
            index=cases*self.samples+(u*self.samples).to(torch.int64)
            x=torch.nn.functional.one_hot(index,self.cases*self.samples).to(torch.float32)
        else:
            if backend=="jax":
                import jax.numpy as xp
            else:xp=np
            cases=xp.asarray(cases);u=xp.asarray(uniform)
            valid=xp.isfinite(cases)&xp.isfinite(u)&(cases==xp.floor(cases))&(cases>=0)&(cases<self.cases)&(u>=0)&(u<1)
            index=cases.astype("int32")*self.samples+(u*self.samples).astype("int32")
            x=xp.eye(self.cases*self.samples,dtype=xp.float32)[index]
        result=self.regressor.predict(x)[0]
        return xp.where(valid[:,None],result,xp.nan) if backend=="jax" else result

    def save(self,path):self.regressor.save(path)

    @classmethod
    def load(cls,path,*,backend="jax",device="cpu"):
        return cls(FrozenRegressor.load(path,backend=backend,device=device))
