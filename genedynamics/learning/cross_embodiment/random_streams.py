"""Named, stateless random stream contract; explicit arrays cross providers."""
from enum import IntEnum


class RandomStream(IntEnum):
    PRIOR_MODE=101
    PRIOR_SKILL=103
    TRANSPORT=107
    TRAIN_SAMPLE=109
    PROBE=113


def jax_normal(seed,sample_ids,window,stream,shape):
    import jax
    import jax.numpy as jp
    key=jax.random.fold_in(jax.random.PRNGKey(seed),int(stream))
    def one(identity):
        sample=jax.random.fold_in(key,identity)
        return jax.random.normal(jax.random.fold_in(sample,window),shape,dtype=jp.float32)
    return jax.vmap(one)(sample_ids)


def jax_modes(seed,sample_ids,probabilities):
    import jax
    import jax.numpy as jp
    key=jax.random.fold_in(jax.random.PRNGKey(seed),int(RandomStream.PRIOR_MODE))
    uniform=jax.vmap(lambda identity:jax.random.uniform(jax.random.fold_in(key,identity),(),dtype=jp.float32))(sample_ids)
    cumulative=jp.cumsum(probabilities)
    return jp.minimum(jp.sum(uniform[:,None]>=cumulative[None,:],axis=-1),probabilities.shape[0]-1).astype(jp.int32)
