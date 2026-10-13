"""Public geometric contact teachers, kept separate from learned inference."""
import numpy as np


def contact_teacher(config,state,elapsed,mode=0,*,recovery=False):
    command=np.zeros((len(state.sample_id),10,13),np.float32)
    position=np.asarray(state.reference_position)
    if config.insertion and recovery and elapsed<.6:
        command[...,12]=config.desired_force  # establish the lip jam first
    elif config.insertion and recovery and elapsed<2.4:
        command[...,2]=.015
        command[...,5]=-.12/1.8
        if mode==1 and elapsed<1.4: command[...,1]=-.012
    else:
        command[...,12]=config.desired_force
        if config.insertion:
            command[...,:2]=np.clip(2.*(np.array(config.goal[:2])-position[:,:2]),-.02,.02)[:,None]
        elif elapsed>1.:
            waypoint=np.broadcast_to(np.array(config.goal[:2]),(len(position),2)).copy()
            if elapsed<3.: waypoint[:,1 if mode==0 else 0]=(-.20 if mode==0 else .38)
            command[...,:2]=np.clip(2.*(waypoint-position[:,:2]),-.04,.04)[:,None]
    return command


def classify_contact_modes(position,force,valid,*,insertion):
    """Actual trace evidence; unknown=-1, no requested-label shortcut."""
    b=position.shape[0]; p=position.reshape(b,-1,3); f=force.reshape(b,-1); v=valid.reshape(b,-1)
    result=np.full(b,-1,np.int32)
    for i in range(b):
        points=p[i,v[i]]; loads=f[i,v[i]]
        if len(points)<2: continue
        if insertion:
            # Recovery requires an actual lip load followed by withdrawal.
            touched=np.flatnonzero(loads>.3)
            if not len(touched): continue
            first=touched[0]
            withdrawn=np.flatnonzero(points[first:,2]>points[first,2]+.012)
            if not len(withdrawn): continue
            until=first+withdrawn[0]
            result[i]=1 if np.min(points[:until+1,1])<points[0,1]-.004 else 0
        else:
            dx=points[:,0]-points[0,0]; dy=points[:,1]-points[0,1]
            x=np.flatnonzero(dx>.015); y=np.flatnonzero(dy>.008)
            if len(x) and len(y): result[i]=0 if x[0]<y[0] else 1
    return result
