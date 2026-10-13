"""Global fractional CVaR and two-pass, padded microbatch gradient accumulation."""
import math
import jax
import jax.numpy as jnp


def cvar_weights(risk,valid,tail_fraction):
    """Uniform subgradient at ties, exact fractional tail mass, no sorting bias."""
    if not 0<tail_fraction<=1: raise ValueError("tail_fraction must be in (0,1]")
    valid=valid.astype(bool)
    count=valid.sum()
    mass=jnp.maximum(count*tail_fraction,1e-8)
    sorted_risk=jnp.sort(jnp.where(valid,risk,-jnp.inf))[::-1]
    index=jnp.clip(jnp.ceil(mass).astype(jnp.int32)-1,0,risk.size-1)
    threshold=sorted_risk[index]
    above=valid&(risk>threshold); tied=valid&(risk==threshold)
    remaining=jnp.maximum(mass-above.sum(),0.)
    weights=(above.astype(risk.dtype)+tied.astype(risk.dtype)*remaining/jnp.maximum(tied.sum(),1))/mass
    return jnp.where(count>0,weights,0.)


def microbatch_value_and_grad(loss_fn,parameters,batch,valid,*,microbatch_size,tail_fraction=.2,multiplier=1.):
    """loss_fn(params,batch)->(per-world objective, [world,ensemble] risk).

    First pass computes the global tail over ALL valid worlds and worst-case
    ensemble members. Second pass recomputes microbatches and accumulates the
    weighted VJP. Adam is applied once by the caller. No host tensor conversion
    or per-environment Python execution is used; static padding is masked.
    """
    n=valid.shape[0]
    if microbatch_size<1: raise ValueError("microbatch size must be positive")
    count=math.ceil(n/microbatch_size); padded=count*microbatch_size
    pad=lambda x:jnp.pad(x,((0,padded-n),)+((0,0),)*(x.ndim-1)).reshape(count,microbatch_size,*x.shape[1:])
    batches=jax.tree.map(pad,batch); masks=pad(valid)
    def first(_,values):
        obj,risk=loss_fn(parameters,values)
        return None,(obj,jnp.max(risk,axis=-1))
    _,(objectives,risks)=jax.lax.scan(first,None,batches)
    weights=jax.lax.stop_gradient(cvar_weights(risks.ravel(),masks.ravel(),tail_fraction).reshape(count,microbatch_size))
    denominator=jnp.maximum(valid.sum(),1)
    def local(p,values,mask,weight):
        objective,risk=loss_fn(p,values)
        return jnp.sum(objective*mask)/denominator+multiplier*jnp.sum(jnp.max(risk,axis=-1)*weight)
    zero=jax.tree.map(jnp.zeros_like,parameters)
    def second(accumulator,values):
        values,mask,weight=values
        gradient=jax.grad(local)(parameters,values,mask,weight)
        return jax.tree.map(jnp.add,accumulator,gradient),None
    gradient,_=jax.lax.scan(second,zero,(batches,masks,weights))
    value=jnp.sum(objectives*masks)/denominator+multiplier*jnp.sum(risks*weights)
    return value,gradient
