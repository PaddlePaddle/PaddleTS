#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Setup script.
"""

from setuptools import find_packages, setup
from pathlib import Path
import paddlets

def read_requirements(path):
    """read requirements"""
    return list(Path(path).read_text().splitlines())

base_reqs = read_requirements("requirements/core.txt")
paddle_reqs = read_requirements("requirements/paddle.txt")
autots_reqs = read_requirements("requirements/autots.txt")

all_reqs = base_reqs + paddle_reqs + autots_reqs

setup(
    name='paddlets',
    version=paddlets.__version__,
    maintainer='paddlets Team',
    maintainer_email='paddlets@baidu.com',
    packages=find_packages(),
    url='https://github.com/PaddlePaddle/PaddleTS',
    license='LICENSE',
    description='PaddleTS (Paddle Time Series Tool), \
           PaddlePaddle-based Time Series Modeling in Python',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    install_requires=base_reqs,
    extras_require={
        "paddle": paddle_reqs,
        "autots": autots_reqs,
        "all": all_reqs
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ]
)
