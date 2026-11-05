#!/usr/bin/env python3
"""Prepare declared moving-fixture response data without physics or training."""
import argparse
import json
from pathlib import Path

from genedynamics.solvers.single.cross_ebdm.learning.response import (
    PublicResponseSpec, prepare_fixture_response_data,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config.get('version') != 'fixture-response-data-preparation-v1':
        raise ValueError('explicit fixture response data recipe required')
    root=Path(__file__).resolve().parents[3]
    entries=[dict(entry,path=root/entry['path']) for entry in config['trajectories']]
    report=prepare_fixture_response_data(entries,root/config['output'],
        spec=PublicResponseSpec(**config['response_spec']),
        segment_steps=config['segment_steps'],calibration_context=config['calibration_context'])
    print(json.dumps({key:report[key] for key in (
        'response_schema','fit_windows','validation_windows','model_trained','new_physics_steps')},indent=2))


if __name__=='__main__':
    main()
