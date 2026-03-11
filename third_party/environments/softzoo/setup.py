from setuptools import setup, find_packages


project_name = 'softzoo'
version = '0.0.1'
classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Simulation :: Soft Robot',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
]
install_requires = [] # check README

setup(name=project_name,
      packages=[project_name.lower()],
      package=find_packages(include=[project_name.lower()]),
      version=version,
      description='SoftZoo: A Soft Robot Co-design Benchmark For Locomotion In Diverse Environments',
      author='Tsun-Hsuan Wang',
      author_email='tsunw@mit.edu',
      install_requires=install_requires,
      keywords=['simulation', 'soft robot'],
      license='MIT',
      include_package_data=True,
      classifiers=classifiers)
