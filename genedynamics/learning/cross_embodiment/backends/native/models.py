"""Host CPU C++ provider. Build is explicit; unsupported training fails early."""
import ctypes
import os
from pathlib import Path
import numpy as np


class NativeModel:
    def __init__(self,spec,device):
        if device not in {"cpu","cpu:0"}: raise ValueError("native v1 supports CPU inference only")
        self.spec=spec
        path=os.environ.get("CROSS_EMBODIMENT_NATIVE_LIBRARY")
        if not path: raise RuntimeError("set CROSS_EMBODIMENT_NATIVE_LIBRARY to the built ABI v1 library")
        self.library=ctypes.CDLL(str(Path(path).resolve()))
        self.library.cross_embodiment_abi_version.restype=ctypes.c_int
        if self.library.cross_embodiment_abi_version()!=1: raise ValueError("unsupported native ABI")
        fptr=ctypes.POINTER(ctypes.c_float); iptr=ctypes.POINTER(ctypes.c_int32)
        self.library.cross_embodiment_mlp.argtypes=[fptr,fptr,iptr,ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,fptr,fptr]
        self.library.cross_embodiment_mlp.restype=ctypes.c_int

    def predict(self,parameters,inputs):
        sizes=(self.spec.input_dim,*self.spec.hidden,self.spec.output_dim)
        if len(parameters)!=len(sizes)-1 or any(
            w.shape!=(self.spec.ensemble,n,m) or b.shape!=(self.spec.ensemble,m) or
            not np.isfinite(w).all() or not np.isfinite(b).all()
            for (w,b),n,m in zip(parameters,sizes[:-1],sizes[1:])):
            raise ValueError("native parameters must match finite model layers")
        x=np.ascontiguousarray(inputs,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=self.spec.input_dim or len(x)<1 or not np.isfinite(x).all():
            raise ValueError("native input must be finite [B,input_dim]")
        weights=np.concatenate([v[0].ravel() for v in parameters]).astype(np.float32)
        biases=np.concatenate([v[1].ravel() for v in parameters]).astype(np.float32)
        dims=np.array([self.spec.input_dim,*self.spec.hidden,self.spec.output_dim],np.int32)
        output=np.empty((self.spec.ensemble,len(x),self.spec.output_dim),np.float32)
        ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        error=self.library.cross_embodiment_mlp(ptr(weights),ptr(biases),dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            len(dims)-1,self.spec.ensemble,len(x),ptr(x),ptr(output))
        if error: raise RuntimeError(f"native inference error {error}")
        return output

    def import_parameters(self,tensors):
        sizes=(self.spec.input_dim,*self.spec.hidden,self.spec.output_dim)
        expected={f"layer.{i}.{name}" for i in range(len(sizes)-1) for name in ("weight","bias")}
        if set(tensors)!=expected: raise ValueError("model layer manifest mismatch")
        layers=[]
        for i,(n,m) in enumerate(zip(sizes[:-1],sizes[1:])):
            w,b=tensors[f"layer.{i}.weight"],tensors[f"layer.{i}.bias"]
            if w.shape!=(self.spec.ensemble,n,m) or b.shape!=(self.spec.ensemble,m):
                raise ValueError("model layer shape mismatch")
            layers.append((np.ascontiguousarray(w,dtype=np.float32),np.ascontiguousarray(b,dtype=np.float32)))
        return tuple(layers)

    @staticmethod
    def synchronize(value): pass
    @staticmethod
    def export_tensors(tensors): return tensors
    @staticmethod
    def initialize(seed): raise NotImplementedError("native v1 inference only")
    @staticmethod
    def train_step(*args): raise NotImplementedError("native v1 inference only")
    @staticmethod
    def train_chunk(*args): raise NotImplementedError("native v1 inference only")
    @staticmethod
    def input_gradient(*args): raise NotImplementedError("native v1 inference only")
