"""Public development excitation records; never query a hidden target model."""
import argparse
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment import ExecutionSpec,MechanismSpec,ObstacleTransferConfig
from genedynamics.learning.cross_embodiment.data import make_episode_plan,collect_dataset
from genedynamics.learning.cross_embodiment.reference import reference_commands


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("tmp/cross_embodiment/response_excitations"))
    args=parser.parse_args()
    def command(task,skill,route,steps):
        value=reference_commands(task,skill,route,steps)
        rng=np.random.default_rng(50+(route=="side"))
        offsets=rng.uniform(-1,1,(steps//10,12)).repeat(10,axis=0)
        value[:,:12]+=offsets*np.array([.005]*3+[.05]*3+[.3]*6)
        return value
    episodes=make_episode_plan(seed=0,robots=("panda","xarm7"),mechanisms=(
        MechanismSpec(),MechanismSpec(.8,2,.8),MechanismSpec(1.15,1,.35),MechanismSpec(.9,4,1.)))
    _,meta=collect_dataset(episodes,role="response_development",execution=ExecutionSpec(backend="numpy",physics_backend="mujoco"),
        task=ObstacleTransferConfig(timeout_seconds=30.),command_factory=command,output=args.output)
    print({k:meta[k] for k in ("episodes_attempted","status_counts","valid_control_steps")})


if __name__=="__main__": main()
