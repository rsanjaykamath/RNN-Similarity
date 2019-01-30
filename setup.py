from setuptools import setup, find_packages
import sys

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='rnn-similarity',
    version='1.0',
    description='RNN-Similarity model for answer sentence selection',
    python_requires='>=3.5',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
)
