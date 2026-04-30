from setuptools import setup, find_packages
from codecs import open
from os import path


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        raw = str(line).strip()
        # Skip comments, blanks, and pip options (e.g. --extra-index-url)
        if not raw or raw.startswith("#") or raw.startswith("-"):
            continue
        requires_list.append(raw)


setup(
    name='genedynamics',
    version="0.0.1",
    description='',
    author='hector',
    author_email='hectorh@cmu.edu',
    packages=find_packages(),
    install_requires=requires_list,
    extras_require={
        # JAX-native kinematics + 3D mesh SDF for serial-arm / contact tasks
        # (genedynamics.envs.robots.jax_kinematics, envs.obstacles.sdf_grid_3d).
        "manipulator": [
            "jaxlie>=1.4",
            "urchin>=0.0.27",
            "rtree>=1.0",  # trimesh.proximity.signed_distance backend
        ],
    },
    entry_points={
        "console_scripts": [
            "genedynamics-deploy=genedynamics.deploy.runner:main",
        ],
    },
)