#!/usr/bin/env python3
"""Paper entry point for semantic corner-guide CPU source qualification."""
import argparse
from pathlib import Path

from genedynamics.tasks.transfer.corner_guide_qualification import qualify_corner_guide


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path(
        "configs/arm/guided_contact_alignment/validation/corner_guided_seating_cpu.json"))
    args=parser.parse_args()
    print(qualify_corner_guide(args.config))


if __name__=="__main__":main()
