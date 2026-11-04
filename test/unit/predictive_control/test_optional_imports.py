"""Selecting a CPU data path must not require optional accelerators."""

import subprocess
import sys


def test_imports_without_jax_or_torch():
    source = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','jax','jaxlib'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
from genedynamics.core.backends.execution import ExecutionStep
from genedynamics.core.backends.differentiation import make_runtime
from genedynamics.learning.datasets import EpisodeStore
from genedynamics.solvers.single.budgeted_repair import ControlProblem
from genedynamics.experiments.framework import BaseRunner
assert make_runtime('numpy').name == 'numpy'
'''
    subprocess.run([sys.executable, "-c", source], check=True)
