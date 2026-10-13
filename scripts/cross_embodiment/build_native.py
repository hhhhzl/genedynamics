"""Explicit CPU ABI build; no compilation or subprocess during inference."""
import argparse
from pathlib import Path
import subprocess
import hashlib
import json
import platform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler",default="g++")
    parser.add_argument("--output",type=Path,default=Path("tmp/cross_embodiment/libcross_embodiment.so"))
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[2]/"genedynamics/learning/cross_embodiment/backends/native/ensemble.cpp"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([args.compiler,"-std=c++17","-O2","-fPIC","-shared",str(source),"-o",str(args.output)],check=True)
    manifest={"abi_version":1,"scope":"CPU inference-only","platform":platform.platform(),
        "compiler":subprocess.check_output([args.compiler,"--version"],text=True).splitlines()[0],
        "flags":["-std=c++17","-O2","-fPIC","-shared"],"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest(),"library":str(args.output.resolve())}
    args.output.with_suffix(args.output.suffix+".json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
