from setuptools import setup, find_packages

from codecs import open
from os import path

setup(
    name='PyMSS',
    version='0.1a1',
    description='Python Magnetostatic Simulation',
    long_description='Library for solving the magnetisation and resulting magnetic flux density of an initial distribution of polyhedra carrying a uniform current density and magnetisation.',
    url='https://github.com/gcassella/PyMSS',
    author='Gino Cassella',
    author_email='gwmcassella@gmail.com',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy'],
)