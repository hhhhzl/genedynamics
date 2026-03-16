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
    entry_points={
        "console_scripts": [
            "genedynamics-deploy=genedynamics.deploy.cli:main",
            "genedynamics-deploy-sim=genedynamics.deploy.sim_plan.sim_process:main",
            "genedynamics-deploy-plan=genedynamics.deploy.sim_plan.plan_process:main",
            "genedynamics-deploy-sim2sim=genedynamics.deploy.sim_plan.sim2sim_launcher:main",
        ],
    },
)