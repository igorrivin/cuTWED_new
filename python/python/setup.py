#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='cutwed',
    version='3.0.0',
    description='A linear memory CUDA algorithm for solving Time Warp Edit Distance',
    author='cuTWED Contributors',
    author_email='igorrivin@gmail.com',
    url='https://github.com/garrettwrong/cuTWED',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
    ],
    extras_require={
        'gpu': ['cupy>=10.0.0'],
        'dev': [
            'pytest>=6.0.0',
            'flake8>=4.0.0',
            'black>=22.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
    python_requires='>=3.7',
)