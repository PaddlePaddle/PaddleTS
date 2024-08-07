#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Setup script.
"""

from setuptools import find_packages, find_namespace_packages, setup
from pathlib import Path


def read_requirements(path):
    """read requirements"""
    return list(Path(path).read_text(encoding='utf-8').splitlines())


all_reqs = read_requirements("requirements.txt")

setup(
    name='paddlets',
    version='1.1.0',
    maintainer='paddlets Team',
    maintainer_email='paddlets@baidu.com',
    packages=find_packages(include=['paddlets', 'paddlets.*']) +
    find_namespace_packages(include=['paddlets', 'paddlets.*']),
    url='https://github.com/PaddlePaddle/PaddleTS',
    license='LICENSE',
    description='PaddleTS (Paddle Time Series Tool), \
           PaddlePaddle-based Time Series Modeling in Python',
    long_description=open(
        'README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    install_requires=all_reqs,
    extras_require={"all": all_reqs},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ])
