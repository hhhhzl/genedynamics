"""Bounded external GPU-memory observation for the P2 model worker."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import time

from genedynamics.experiments.artifacts import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--seconds',type=int,default=7200);args=p.parse_args()
    start=time.monotonic();peak=0.;device_peak=0.;records=[];samples=0;last=0.
    while time.monotonic()-start<args.seconds and not args.stop.exists():
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
        own=[]
        for row in rows:
            pid,memory,uuid=[v.strip() for v in row.split(',')]
            if uuid!=args.gpu_uuid:continue
            try:command=Path('/proc')/pid/'cmdline';command=command.read_bytes().decode().replace('\0',' ')
            except (FileNotFoundError,PermissionError):continue
            if 'scripts.cross_embodiment.gpu_finite_collection' in command:
                own.append(dict(pid=int(pid),memory_MiB=float(memory)));peak=max(peak,float(memory))
        device=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
        values=None
        for row in device:
            uuid,used,total,util=[v.strip() for v in row.split(',')]
            if uuid==args.gpu_uuid:
                values=dict(memory_used_MiB=float(used),memory_total_MiB=float(total),utilization_percent=float(util))
                device_peak=max(device_peak,float(used))
        samples+=1
        if own and time.monotonic()-last>=10:
            records.append(dict(elapsed_seconds=time.monotonic()-start,workers=own,device=values));last=time.monotonic()
        if samples%10==0 or own:
            atomic_json(args.output,dict(complete=False,samples=samples,worker_peak_MiB=peak,device_peak_MiB=device_peak,
                gpu_uuid=args.gpu_uuid,records=records,scope='1 Hz external observation; sampled peak, not allocator high-water guarantee'))
        time.sleep(1.)
    atomic_json(args.output,dict(complete=True,samples=samples,worker_peak_MiB=peak,device_peak_MiB=device_peak,
        gpu_uuid=args.gpu_uuid,records=records,analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stopped_by_marker=args.stop.exists(),scope='1 Hz external observation; sampled peak, not allocator high-water guarantee'))


if __name__=='__main__':main()
