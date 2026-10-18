"""Causal batching/reorder/reset and continuation through public guide events."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.guide_pose_feedback import PublicGuidePoseFeedbackPolicy
from test.transfer_methods.test_public_observed_response import record


def test_guide_pose_public_memory_continuation_reorder_and_reset():
    template,_=record(); public=[]
    for mode in (0,1):
        p={k:np.repeat(v[:1],60,axis=0) for k,v in template.items()}
        p['elapsed_seconds'][:,0]=np.arange(60)*.02;p['measured_wrench'][:,2]=5.
        p['tcp_position'][:,:2]=[.38,-.20]
        p['tcp_position'][20:,:2]=[.46,-.20] if mode==0 else [.38,-.18]
        p['tcp_position'][40:,:2]=[.46,-.18]
        public.append(p)
    spec=task_spec('T2',robot='panda'); batch=PublicGuidePoseFeedbackPolicy(spec,modes=(0,1)); solo=[PublicGuidePoseFeedbackPolicy(spec,modes=(i,)) for i in (0,1)]
    for tick in range(min(len(p['tcp_position']) for p in public)):
        obs=[{k:v[tick:tick+1] for k,v in p.items()} for p in public]; combined={k:np.concatenate([p[k] for p in obs]) for k in obs[0]}
        before=batch.get_state(); first=batch.act(combined); after=batch.get_state(); batch.restore(before)
        assert np.array_equal(batch.act(combined),first)
        batch.restore(after); assert np.array_equal(batch.act(combined),first)
        assert np.array_equal(first,np.concatenate([actor.act(p) for actor,p in zip(solo,obs)]))
        batch.reorder([1,0]); reverse={k:v[[1,0]] for k,v in combined.items()}; assert np.array_equal(batch.act(reverse),first[[1,0]])
        batch.reorder([1,0])
    batch.reset(modes=(0,1)); combined={k:np.concatenate([p[k][:1] for p in public]) for k in public[0]}
    fresh=PublicGuidePoseFeedbackPolicy(spec,modes=(0,1))
    assert np.array_equal(batch.act(combined),fresh.act(combined))
