from setuptools import find_packages
from setuptools import setup

setup(
    name='retargeting',
    version='0.3.2',
    packages=find_packages(
        include=('retargeting', 'retargeting.*')),
)
