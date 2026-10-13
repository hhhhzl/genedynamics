"""Lossless outer compression of completed evidence, preserving original SHA.

Run only on an idle experiment directory. The committed metadata and raw NPZ
bytes remain unchanged; load_bundle verifies the original SHA after decoding.
"""
import argparse
import gzip
import hashlib
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("--exclude",action="append",default=[])
    a=p.parse_args();saved=0;count=0
    lock=a.root/".compression.lock"
    handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(handle)
    for source in a.root.rglob("tensors-*.npz"):
        if any(part in a.exclude for part in source.relative_to(a.root).parts):continue
        raw=source.read_bytes();packed=gzip.compress(raw,compresslevel=6,mtime=0)
        if len(packed)>=len(raw):continue
        target=source.with_suffix(source.suffix+".gz");temporary=target.with_suffix(".new")
        temporary.write_bytes(packed)
        if hashlib.sha256(gzip.decompress(temporary.read_bytes())).digest()!=hashlib.sha256(raw).digest():
            raise RuntimeError("compression checksum mismatch; original preserved")
        os.replace(temporary,target);source.unlink();saved+=len(raw)-len(packed);count+=1
    lock.unlink()
    print(dict(files=count,bytes_saved=saved,original_npz_sha256_preserved=True),flush=True)


if __name__=="__main__":main()
