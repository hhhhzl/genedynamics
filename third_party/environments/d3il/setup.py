from setuptools import setup, find_packages

setup(
    name="environments.d3il.d3il_sim",
    version="0.2",
    description="Franka Panda Simulators",
    license="MIT",
    package_data={"models": ["*"]},
    packages=find_packages(),
)
