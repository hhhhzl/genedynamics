"""
Third-party environment packages.

We vendor external environment code here (or pin it via git submodules) so that
`enerdynamics` can integrate with them without copying their sources into the
core package.

D3IL expects imports like `environments.d3il.*`, so `third_party` must be on
PYTHONPATH for those imports to resolve.
"""


