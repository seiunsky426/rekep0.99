#!/usr/bin/env python3

# Source classification: NEW safety adapter.

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["rekpiper_mapping"],
    package_dir={"": "src"},
)

setup(**setup_args)
