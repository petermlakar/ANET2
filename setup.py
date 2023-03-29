import os
from setuptools import setup, find_packages

def parse_requirements(file):
    return [line for line in open(os.path.join(os.path.dirname(__file__), file))]

setup(
    name = "ANET2",
    python_requires = ">=3.6",
    version = "2.0",
    description = "AtmosphereNET 2.0 post-processing of ensemble weather forecasts",
    author = "Peter Mlakar",
    author_email = "pm4824@student.uni-lj.si",
    packages=find_packages(include=["hidra", 'hidra.*']),
    install_requires=parse_requirements('requirements.txt')
)
