"""Finite-policy confidence, probe separation and supported mode reweighting."""
import numpy as np


class NumpyResearchAudit:
    @staticmethod
    def paired_improvement(actual,alpha):
        a=np.asarray(actual,float)
        if a.ndim!=2 or len(a)<1 or a.shape[1]<2 or not 0<alpha<1 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:
            raise ValueError("independent reset clusters of predeclared [0,1] costs required")
        n,m=a.shape
        differences=a[:,0,None]-a[:,1:]
        radius=float(np.sqrt(2*np.log((m-1)/alpha)/n))
        means=differences.mean(0);lower=means-radius
        return {"clusters":n,"alpha":alpha,"baseline_column":0,"mean_improvements":means.tolist(),
                "one_sided_radius":radius,"improvement_lower":lower.tolist(),"certified_vs_column0":(lower>0).tolist(),
                "scope":"one-sided Hoeffding union bound on paired differences in [-1,1]; iid reset clusters, frozen policies and cost; conditional on calibration; no model-fidelity or cross-calibration guarantee"}

    @staticmethod
    def separation(signatures,noise,bias):
        z=np.asarray(signatures,float)
        if z.ndim!=2 or len(z)<2 or not np.isfinite(z).all() or min(noise,bias)<0 or not np.isfinite([noise,bias]).all():
            raise ValueError("finite signatures and nonnegative declared error bounds required")
        distances=np.linalg.norm(z[:,None]-z[None],axis=-1)
        np.fill_diagonal(distances,np.inf)
        minimum=float(distances.min()); threshold=float(2*(noise+bias))
        return {"minimum_separation":minimum,"error_threshold":threshold,"identified":minimum>threshold,
                "scope":"declared finite mechanism class; true mechanism must belong; error bounds externally justified"}

    @staticmethod
    def cost_certificate(actual,predicted,alpha):
        """Rows are independent paired RESET CLUSTERS, columns frozen policies.

        Costs must be predeclared and bounded in [0,1]. Union Hoeffding bounds
        cover both finite model cost estimation and paired model bias. This
        often returns no certificate with small n; empirical success is not a
        substitute. Independence and validation isolation are caller duties.
        """
        a,p=map(lambda x:np.asarray(x,float),(actual,predicted))
        if a.ndim!=2 or a.shape!=p.shape or min(a.shape)<1 or not 0<alpha<1:
            raise ValueError("paired [independent_clusters,frozen_policies] costs required")
        if not np.isfinite(a).all() or not np.isfinite(p).all() or min(a.min(),p.min())<0 or max(a.max(),p.max())>1:
            raise ValueError("predeclared bounded [0,1] cost required")
        n,m=a.shape
        model_sampling=float(np.sqrt(np.log(4*m/alpha)/(2*n)))
        bias_radius=float(np.sqrt(np.log(2*m/alpha)/(2*n)))
        epsilon=min(1.,float(np.abs(a-p).mean(0).max())+bias_radius)
        gap=p.mean(0)[0]-p.mean(0)
        conservative=gap-2*epsilon-2*model_sampling
        return {"clusters":n,"frozen_policies":m,"alpha":alpha,"model_mean_radius":model_sampling,
                "uniform_cost_error_upper":epsilon,"model_gap_vs_column0":gap.tolist(),
                "true_improvement_lower":conservative.tolist(),"certified_vs_column0":(conservative>0).tolist(),
                "scope":"finite frozen policies under declared iid reset distribution, conditional on calibration; no global kernel certificate"}

    @staticmethod
    def mode_law(costs,weights,feasible,temperature):
        c,w=np.asarray(costs,float),np.asarray(weights,float); ok=np.asarray(feasible,bool)
        if c.ndim!=1 or c.shape!=w.shape or ok.shape!=c.shape or not np.isfinite(c).all() or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1) or not np.isfinite(temperature) or temperature<=0:
            raise ValueError("finite costs and normalized source law required")
        support=ok&(w>0)
        if not support.any():
            return {"probabilities":np.zeros_like(w).tolist(),"unsupported":True,"stop_mass":1.}
        log=np.full_like(c,-np.inf);log[support]=np.log(w[support])-c[support]/temperature
        p=np.exp(log-log.max());p/=p.sum()
        return {"probabilities":p.tolist(),"unsupported":False,"stop_mass":0.,
                "scope":"finite supported source modes, KL-regularized model costs; feasibility must be budget-supported, not evaluation oracle"}
