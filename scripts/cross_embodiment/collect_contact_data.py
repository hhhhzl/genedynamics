"""Collect episode-grouped real contact data from public development mechanisms."""
import argparse
from pathlib import Path
from genedynamics.learning.cross_embodiment.contact_data import collect_contact_dataset
from genedynamics.learning.cross_embodiment import MechanismSpec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",default="guided_contact_alignment")
    parser.add_argument("--role",choices=["source_demonstrations","response_development"],default="source_demonstrations")
    parser.add_argument("--robots",nargs="+",default=None)
    parser.add_argument("--excite",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    public=args.role=="response_development"
    _,meta=collect_contact_dataset(args.output,environment=args.task,role=args.role,
        robots=tuple(args.robots or (["panda","xarm7","ur5e"] if public else ["panda"])),
        mechanisms=(MechanismSpec(),MechanismSpec(.8,2,.8),MechanismSpec(1.15,1,.35),MechanismSpec(.9,4,1.)) if public else (MechanismSpec(),),
        excite=args.excite)
    print({k:meta[k] for k in ("environment","role","episodes_attempted","successes","observed_modes","valid_steps")})


if __name__=="__main__": main()
