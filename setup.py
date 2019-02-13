from typing import List
from setuptools import setup, find_packages

required = []  # type: List[str]

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('doc/README.rst') as f:
    long_description = f.read()

setup(name='npeet',
      version='1.0.2',
      description='Non-parametric Entropy Estimation Toolbox',
      author='Greg Ver Steeg',
      author_email='gregv@isi.edu',
      url='https://github.com/MaxwellRebo/NPEET',
      packages=find_packages(),
      install_requires=required,
      tests_require=['pytest'],
      long_description=long_description,
      )
