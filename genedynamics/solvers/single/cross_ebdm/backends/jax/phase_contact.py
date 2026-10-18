"""Fused phase-prior/affine-transport/feedback/MJX execution.

The reference prior and controller match the finite CPU study. Random streams
are supplied at the boundary, so batching cannot change an episode's draw.
This is not the older online source-query/neural-response policy.
"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jp

from genedynamics.envs.domains.manipulation.transfer_workcell.backends.jax.execution import JaxExecutionBackend
from genedynamics.envs.domains.manipulation.transfer_workcell.backends.jax.contact import _contact_scope
from genedynamics.tasks.transfer.behavior import behavior_reset, advance_behavior, keyway_alignment, reference_commands
from genedynamics.tasks.transfer.runtime import RandomRequest
from genedynamics.solvers.single.cross_ebdm.learning.artifact_bundle import load_bundle


class PhaseChunk(NamedTuple):
    state: object
    memory: object
    trace: object                 # [B, K*10, ...]
    phases: object                # [K, B]
    references: object            # [K, B, 5]
    raw_commands: object          # [B, K*10, 13]
    events: object                # [B, K*10] diagnostic side channel


def affine_bank(paths):
    """Canonical 16-feature affine maps, retaining portable artifact semantics."""
    weights, biases = [], []
    for path in paths:
        weight = np.zeros((16, 13), np.float32)
        if path is None:
            weight[:13] = np.eye(13, dtype=np.float32)
            bias = np.zeros(13, np.float32)
        else:
            tensors, meta = load_bundle(path)
            spec = meta['model_spec']; width = spec['input_dim']
            if (meta.get('learned_role') != 'reference_transport' or spec['hidden'] or
                    spec['ensemble'] != 1 or spec['output_dim'] != 13 or width not in (13, 15, 16)):
                raise ValueError('phase GPU execution requires a declared portable affine reference transport')
            for name in ('input', 'output'):
                if (np.any(tensors[f'normalization.{name}_center'] != 0) or
                        np.any(tensors[f'normalization.{name}_scale'] != 1)):
                    raise ValueError('affine reference transport must have identity normalization')
            original = tensors['layer.0.weight'][0]
            weight[:width] = original
            if width == 15:
                weight[15] = original[14]; weight[14] = 0.
            bias = tensors['layer.0.bias'][0]
        weights.append(weight); biases.append(bias)
    if not weights:
        raise ValueError('nonempty affine bank required')
    return np.stack(weights), np.stack(biases)


def replay_noise(seeds, windows):
    """Exactly the CPU prior's per-episode (seed, control step) random stream."""
    return np.stack([np.stack([np.random.default_rng(np.random.SeedSequence([int(seed), k*10]))
                               .normal(size=(1, 5))[0].astype(np.float32) for seed in seeds])
                     for k in range(windows)])


class JaxPhaseContactPolicy:
    def __init__(self, runner, prior):
        r = runner._impl
        if (not hasattr(r, '_chunk_one') or runner.config.mobile or
                runner.config.socket_profile != 'matched_keyway' or prior.regressor.model.backend != 'jax'):
            raise ValueError('phase GPU policy requires fixed matched-keyway JAX physics and a JAX phase prior')
        if prior.regressor.model._impl.device != r.device:
            raise ValueError('prior and physics must reside on the same device')
        self.runner, self.prior = runner, prior
        self._physics=(jax.jit(jax.vmap(lambda model,g,d,s,c,b:r._chunk_events_one(s,c,b,model,g,d)),device=r.device)
                       if r.heterogeneous else jax.jit(jax.vmap(r._chunk_events_one),device=r.device))
        self._run = jax.jit(self._chunk, device=r.device)

    @_contact_scope
    def reset_episodes(self, seeds, modes):
        """Replay the CPU study's individual (seed, 2*seed+mode) resets."""
        if len(seeds)!=len(modes) or not len(seeds):raise ValueError('nonempty paired reset/mode vectors required')
        r=self.runner._impl
        if r.heterogeneous and len(seeds)!=len(r.references):raise ValueError('mechanism/reset batch mismatch')
        ids=[int(s)*2+int(m) for s,m in zip(seeds,modes)]
        if any(i<0 or i>=2**32 for i in ids):raise ValueError('episode identity must fit uint32')
        qpos=np.stack([(r.references[i] if r.heterogeneous else r.reference)
            .reset_batch(RandomRequest(int(seed),(identity,)),initial='jammed').physics[0].qpos
            for i,(seed,identity) in enumerate(zip(seeds,ids))]).astype(np.float64)
        qpos,ids,zero=jax.device_put((qpos,np.asarray(ids,np.uint32),np.zeros_like(qpos)),r.device)
        return r._reset_batched(r.models,qpos,ids,zero) if r.heterogeneous else r._reset_rows(qpos,ids,zero)

    def _chunk(self, state, memory, noise, weights, biases):
        r = self.runner._impl; config = r.config

        def window(carry, draw):
            state, memory = carry
            position = state.physics.site_xpos[:, r.sid]
            rotation = state.physics.site_xmat[:, r.sid]
            aligned = keyway_alignment(position, rotation, config.mating_clearance, xp=jp)
            updated = advance_behavior(memory, position, rotation, state.measured_force, xp=jp,
                                       dt=10*config.control_dt, goal_z=config.goal[2], alignment_mask=aligned)
            updated = jax.tree.map(lambda n,o:jp.where((state.status==0).reshape((-1,)+(1,)*(n.ndim-1)),n,o),updated,memory)
            reference = self.prior.reference(updated, draw)
            commands = reference_commands(reference, state.reference_position, state.reference_rotation,
                position, rotation, xp=jp, dt=config.control_dt, integral_bias=updated.tracking_integral)
            commands=commands.astype(jp.float32)  # portable native affine input contract
            mask = jp.broadcast_to((updated.phase==3)[:,None,None], (*commands.shape[:2],1)).astype(jp.float32)
            align = jp.broadcast_to((updated.phase==2)[:,None,None], mask.shape).astype(jp.float32)
            inputs = jp.concatenate((commands, commands[...,12:13]*mask, align, mask), axis=-1)
            commands = jp.einsum('bli,bio->blo', inputs, weights)+biases[:,None,:]
            body = jp.zeros((*commands.shape[:2],3),jp.float32)
            final,(trace,events) = (self._physics(r.models,r.gains,r.delays,state,commands,body) if r.heterogeneous
                                   else self._physics(state,commands,body))
            return (final, updated), (trace, updated.phase, reference, commands,events)

        (state,memory),(trace,phases,references,commands,events) = jax.lax.scan(window,(state,memory),noise)
        flatten = lambda a:jp.swapaxes(a,0,1).reshape((a.shape[1],a.shape[0]*a.shape[2],*a.shape[3:]))
        return PhaseChunk(state,memory,jax.tree.map(flatten,trace),phases,references,flatten(commands),jax.tree.map(flatten,events))

    @_contact_scope
    def initial_memory(self, modes):
        modes = np.asarray(modes,np.int32)
        if modes.ndim!=1 or not np.isin(modes,(0,1)).all():raise ValueError('modes must be a vector of 0/1')
        memory=behavior_reset(len(modes),modes)
        memory=memory._replace(tracking_integral=memory.tracking_integral.astype(np.float64))
        return jax.device_put(memory,self.runner._impl.device)

    @_contact_scope
    def run_chunk(self,state,memory,noise,weights,biases):
        batch=state.sample_id.shape[0]
        if noise.ndim!=3 or noise.shape[1:]!=(batch,5) or weights.shape!=(batch,16,13) or biases.shape!=(batch,13):
            raise ValueError('noise [K,B,5], affine weights [B,16,13], biases [B,13] required')
        # All inputs are explicit device arrays here. Memory-bounded dispatch
        # slices a whole episode chunk; it never steps worlds on the host.
        size=self.runner._impl.microbatch_size or batch
        results=[]
        for start in range(0,batch,size):
            stop=min(start+size,batch); cut=JaxExecutionBackend._slice_tree
            results.append(self._run(cut(state,start,stop),cut(memory,start,stop),
                jax.lax.slice_in_dim(noise,start,stop,axis=1),cut(weights,start,stop),cut(biases,start,stop)))
        if len(results)==1:return results[0]
        return PhaseChunk(JaxExecutionBackend._join([x.state for x in results]),
                          JaxExecutionBackend._join([x.memory for x in results]),
                          JaxExecutionBackend._join([x.trace for x in results]),
                          jp.concatenate([x.phases for x in results],axis=1),
                          jp.concatenate([x.references for x in results],axis=1),
                          jp.concatenate([x.raw_commands for x in results],axis=0),
                          JaxExecutionBackend._join([x.events for x in results]))
