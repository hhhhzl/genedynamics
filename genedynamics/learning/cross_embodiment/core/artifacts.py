"""Portable named tensors + JSON metadata, with no pickle/framework objects.

NumPy is imported only during file I/O. Tensor conversion is explicitly at this
host artifact boundary, never in a rollout or inference hot path.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from ..contracts import SCHEMA_VERSION


def save_bundle(path, tensors, metadata):
    from genedynamics.experiments.artifacts import artifact_write
    import numpy as np
    path = Path(path)
    arrays = {name: np.asarray(value) for name, value in tensors.items()}
    if any(not name or array.dtype.kind not in "fiub" or not np.isfinite(array).all()
           for name, array in arrays.items()):
        raise ValueError("bundle requires named finite numeric tensors")
    document = dict(metadata, schema_version=SCHEMA_VERSION, linear_weight_layout="in_out",
                    tensor_specs={name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in arrays.items()})
    encoded = json.dumps(document, allow_nan=False)  # fail before writing invalid metadata
    # Deflate can expand incompressible data slightly. Reserve an uncompressed
    # upper bound, ZIP headers and metadata while retaining old generations.
    reserve = sum(a.nbytes + a.nbytes // 1000 + 4096 for a in arrays.values()) + len(encoded.encode()) * 2 + 65536
    with artifact_write(path, reserve):
        _publish(path, arrays, document)


def _publish(path, arrays, document):
    import numpy as np
    path.mkdir(parents=True, exist_ok=True)
    # Content-addressed tensor generations + atomic metadata publication keep
    # the previous checkpoint loadable if the process dies during replacement.
    fd,temp=tempfile.mkstemp(dir=path,prefix=".tensors-",suffix=".npz")
    try:
        with os.fdopen(fd,"wb") as stream:
            np.savez_compressed(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
        digest=hashlib.sha256(Path(temp).read_bytes()).hexdigest()
        generation=f"tensors-{digest}.npz"
        os.replace(temp,path/generation)
        document.update(tensor_sha256=digest,tensor_file=generation)
        fd,meta_temp=tempfile.mkstemp(dir=path,prefix=".metadata-")
        try:
            with os.fdopen(fd,"w") as stream:
                stream.write(json.dumps(document,indent=2,allow_nan=False)+"\n")
                stream.flush(); os.fsync(stream.fileno())
            os.replace(meta_temp,path/"metadata.json")
        finally:
            if os.path.exists(meta_temp): os.unlink(meta_temp)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def load_bundle(path):
    import numpy as np
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("linear_weight_layout") != "in_out":
        raise ValueError("unsupported bundle schema/weight convention")
    name=metadata.get("tensor_file","tensors.npz")  # legacy artifacts remain readable
    if Path(name).name!=name or not name.endswith(".npz"):
        raise ValueError("invalid bundle tensor filename")
    source=path/name
    try:payload=source.read_bytes()
    except FileNotFoundError:
        import gzip
        payload=gzip.decompress((path/(name+".gz")).read_bytes())
    digest = hashlib.sha256(payload).hexdigest()
    if digest != metadata.get("tensor_sha256"):
        raise ValueError("bundle tensor checksum mismatch")
    import io
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        tensors = {name: data[name].copy() for name in data.files}
    if any(a.dtype.kind not in "fiub" or not np.isfinite(a).all() for a in tensors.values()):
        raise ValueError("invalid bundle tensors")
    expected = {name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in tensors.items()}
    if metadata.get("tensor_specs") != expected:
        raise ValueError("bundle tensor shape/dtype manifest mismatch")
    return tensors, metadata
