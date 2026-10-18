"""Native diffusion batches from sealed executed TARGET demonstrations.

Only numerical chart/normalizer helpers are reused from source training. No
target trajectory is relabeled a successful source demonstration.
"""
import numpy as np
import torch

from genedynamics.learning.datasets.transfer.target import PublicTargetTrajectory
from .task_training import SourceDemonstrations,response_chart


def public_target_windows(row, *, history_len=8, horizon=16):
    """Convert a sealed ACP19 target record into native history windows."""
    from genedynamics.tasks.transfer.guide_observation import context_sequence
    from genedynamics.tasks.transfer.specs import task_spec
    from genedynamics.solvers.single.acp.backends import spatial
    from .task_training import PUBLIC_FIELDS, history_features, pose

    if (not isinstance(row, PublicTargetTrajectory)
            or row.metadata["permission"] != "C_supervised"
            or row.metadata["chart"] != "acp_world19"):
        raise ValueError("executed native-controller supervised commands required")
    public = row.public
    count = len(row.requests)
    context = context_sequence(
        public,
        task_spec(row.task, robot=row.metadata["robot"]),
        row.metadata["mode"],
    )
    reference = spatial.pose9_to_SE3(row.requests[:, :9])
    virtual = spatial.pose9_to_SE3(row.requests[:, 9:18])
    observations, actions = [], []
    for step in range(count - horizon + 1):
        history = [
            {key: public[key][index:index + 1] for key in PUBLIC_FIELDS}
            for index in range(max(0, step - history_len + 1), step + 1)
        ]
        observations.append({
            key: value[0]
            for key, value in history_features(
                history,
                history_len,
                controller_context=context[step:step + 1],
            ).items()
        })
        inverse = spatial.SE3_inv(pose(
            public["tcp_position"][step], public["tcp_rotation"][step]
        ))
        indices = np.arange(step, step + horizon)
        actions.append(np.concatenate((
            spatial.SE3_to_pose9(inverse @ reference[indices]),
            spatial.SE3_to_pose9(inverse @ virtual[indices]),
            row.requests[indices, -1:],
        ), axis=-1).astype(np.float32))
    if not observations:
        raise ValueError("demonstration shorter than native prediction horizon")
    return (
        {key: np.stack([value[key] for value in observations]) for key in observations[0]},
        np.stack(actions),
    )


class PublicACPTaskDataset:
    def __init__(self,rows,*,purpose,history_len=8,horizon=16,action_chart="response19"):
        if purpose not in {"fit","validation"} or not rows:raise ValueError("explicit nonempty episode split required")
        observations=[];actions=[];self.receipts=[]
        for row in rows:
            if not isinstance(row,PublicTargetTrajectory) or row.metadata["purpose"]!=purpose:
                raise ValueError("target fit/validation episodes cannot cross the split")
            obs,action=public_target_windows(row,history_len=history_len,horizon=horizon)
            observations.append(obs);actions.append(action)
            self.receipts.append(dict(sha256=row.digest,robot=row.metadata["robot"],mode=row.metadata["mode"],purpose=purpose,controls=len(row.requests)))
        if len({r["sha256"] for r in self.receipts})!=len(rows):raise ValueError("duplicate complete target episode")
        self.observations={k:torch.from_numpy(np.concatenate([v[k] for v in observations])) for k in observations[0]}
        self.native_actions=torch.from_numpy(np.concatenate(actions));self.actions=self.native_actions.clone();self.action_chart=action_chart
        if action_chart=="response19":self.actions=response_chart(self.actions)
        elif action_chart!="native19":raise ValueError("unknown native action chart")

    def __len__(self):return len(self.actions)

    def batch(self,indices):
        return {"obs":{"sparse":{k:v[indices] for k,v in self.observations.items()}},"action":{"sparse":self.actions[indices]}}

    def normalizer(self,*,rotation_residual=True):
        # This method is purely numerical and uses only this object's tensors.
        return SourceDemonstrations.normalizer(self,rotation_residual=rotation_residual)
