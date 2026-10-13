"""Autoregressive response ensemble with full carry and sticky predicted risks."""
import jax
import jax.numpy as jp


def forward_member_inputs(parameters,inputs):
    value=inputs
    for i,(weight,bias) in enumerate(parameters):
        value=jp.einsum("ebi,eio->ebo",value,weight)+bias[:,None]
        if i+1!=len(parameters): value=jp.tanh(value)
    return value


def response_rollout(response,carry,commands,body,context):
    """commands [B,K,H,13] → carry/risk [E,B,K,...], no initial-state reuse.

    Each ensemble member advances its own full carry. Predicted collision or
    contact-loss probabilities freeze subsequent carry, and worst observed
    risk is retained. This is a surrogate rollout, never physical success.
    """
    layout=response.metadata["layout"]; e=response.model.spec.ensemble
    state=jp.broadcast_to(carry,(e,*carry.shape))
    b=carry.shape[0]; d=carry.shape[-1]
    done=jp.zeros((e,b),bool)
    contact=response.metadata.get("risk_kind")=="contact"
    initialrisk=jp.zeros((e,b,3),carry.dtype)
    if not contact: initialrisk=initialrisk.at[...,0].set(.005)
    def step(previous,command):
        state,done,risk=previous
        flat=command.reshape(b,-1)
        feature=jp.concatenate((state,jp.broadcast_to(flat,(e,*flat.shape)),
            jp.broadcast_to(body,(e,*body.shape)),jp.broadcast_to(context,(e,*context.shape))),axis=-1)
        prediction=forward_member_inputs(response.parameters,(feature-response.ic)/response.iscale)*response.oscale+response.oc
        candidate=state+prediction[...,:d]
        # Keep TCP rotations on SO(3); other state features advance as learned.
        start,end=layout.get("semantic_slice",[14,32]); index=start+3
        matrix=candidate[...,index:index+9].reshape(e,b,3,3)
        normalize=lambda v:v/jp.sqrt(jp.maximum(jp.sum(v*v,axis=-1,keepdims=True),1e-12))
        x=matrix[...,0]
        x=normalize(jp.where(jp.sum(x*x,axis=-1,keepdims=True)>1e-12,x,jp.array([1.,0.,0.])))
        y=matrix[...,1]-jp.sum(matrix[...,1]*x,axis=-1,keepdims=True)*x
        basis=jp.eye(3)[jp.argmin(jp.abs(x),axis=-1)]
        fallback=basis-jp.sum(basis*x,axis=-1,keepdims=True)*x
        y=normalize(jp.where(jp.sum(y*y,axis=-1,keepdims=True)>1e-12,y,fallback))
        rotation=jp.stack((x,y,jp.cross(x,y)),axis=-1)
        candidate=candidate.at[...,index:index+9].set(rotation.reshape(e,b,9))
        nextstate=jp.where(done[...,None],state,candidate)
        local=prediction[...,-3:]
        if contact:
            risk=jp.maximum(risk,local)
            done=done|(local[...,1]>.5)|(local[...,2]>.5)|(local[...,0]>25.)
        else:
            risk=jp.stack((jp.minimum(risk[...,0],local[...,0]),jp.maximum(risk[...,1],local[...,1]),jp.maximum(risk[...,2],local[...,2])),axis=-1)
            done=done|(local[...,0]<0.)|(local[...,1]>2.5)
        return (nextstate,done,risk),(nextstate,risk,done)
    _,(states,risks,terminal)=jax.lax.scan(step,(state,done,initialrisk),jp.swapaxes(commands,0,1))
    return jp.transpose(states,(1,2,0,3)),jp.transpose(risks,(1,2,0,3)),jp.transpose(terminal,(1,2,0))
