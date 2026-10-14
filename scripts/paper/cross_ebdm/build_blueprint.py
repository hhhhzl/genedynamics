"""Compile the paper draft with a project-owned, removable TeX cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import io
import tarfile
import urllib.request

from genedynamics.experiments.artifacts import atomic_json, POLICY_ENV
from genedynamics.experiments.process import run_stage
from scripts.research.environment import ROOT


def provision(runtime):
    base=runtime/'tex';compiler=base/'tectonic'
    if compiler.exists():return compiler
    url='https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz'
    expected='dfb82876f2986862996e564fa507a9e576e0c1e3bee63c2c1bd677c2543e6407'
    request=urllib.request.Request(url,headers={'User-Agent':'genedynamics-paper-build'})
    with urllib.request.urlopen(request,timeout=60) as stream:payload=stream.read(64*1024**2+1)
    if len(payload)>64*1024**2 or hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError('pinned compiler archive checksum mismatch')
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        members=[m for m in archive.getmembers() if m.isfile() and Path(m.name).name=='tectonic']
        if len(members)!=1:raise ValueError('unexpected compiler archive')
        binary=archive.extractfile(members[0]).read()
    base.mkdir(parents=True,exist_ok=True);compiler.write_bytes(binary);compiler.chmod(0o755)
    # Runtime setup is separate from the experiment artifact policy; it is
    # removed by the existing owned-environment cleaner.
    (base/'compiler.json').write_text(json.dumps(dict(version='0.15.0',url=url,
        archive_sha256=expected,binary_sha256=hashlib.sha256(binary).hexdigest(),location=str(compiler)))+'\n')
    return compiler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',type=Path)
    p.add_argument('--runtime',type=Path,default=ROOT/'.runtime/cross_ebdm')
    args=p.parse_args();runtime=args.runtime.resolve();output=args.output.resolve()
    if args.policy:os.environ[POLICY_ENV]=str(args.policy.resolve())
    if json.loads((runtime/'.owned-environment.json').read_text())['repository']!=str(ROOT):
        raise ValueError('paper compiler must use this project-owned runtime')
    compiler=provision(runtime);source=ROOT/'latex/cross_ebdm/intervention_calibrated_sim2real_blueprint.tex'
    output.mkdir(parents=True,exist_ok=True)
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.parent.rglob('*.tex')) if 'archive' not in p.parts}
    env=dict(os.environ,XDG_CACHE_HOME=str(runtime/'tex/cache'),XDG_CONFIG_HOME=str(runtime/'tex/config'),
             TMPDIR=str(runtime/'tmp'))
    command=[str(compiler),'--untrusted','--keep-logs','--web-bundle',
             'https://relay.fullyjustified.net/default_bundle_v33.tar','--outdir',str(output),str(source)]
    result=run_stage('blueprint',command,output,env=env,timeout=600)
    if result['status']!='passed':raise RuntimeError('paper draft did not compile')
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha for p,sha in inputs.items()):
        raise RuntimeError('manuscript changed during compilation')
    pdf=output/(source.stem+'.pdf')
    font=Path(subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True))
    proof=dict(compiler=json.loads((runtime/'tex/compiler.json').read_text()),
               bundle='https://relay.fullyjustified.net/default_bundle_v33.tar',inputs=inputs,
               system_cjk_font=dict(path=str(font),sha256=hashlib.sha256(font.read_bytes()).hexdigest()),
               pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pdf=str(pdf),
               warnings='TeX layout warnings remain in the retained engine log; no errors',
               scope='compiled paper draft; build provenance is separate from experimental evidence')
    atomic_json(output/'build-provenance.json',proof)
    print(json.dumps({'pdf':str(pdf),'source_files':len(inputs),'pdf_sha256':proof['pdf_sha256']},indent=2))


if __name__=='__main__':main()
