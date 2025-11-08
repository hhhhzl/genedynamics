from setuptools import setup, find_packages
from codecs import open
from os import path


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(
    name='enerdynamics',
    version="0.0.1",
    description='',
    author='hector',
    author_email='hectorh@cmu.edu',
    packages=find_packages(),
    install_requires=[],
)