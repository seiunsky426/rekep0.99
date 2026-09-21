#!/usr/bin/env python3

# Source classification: NEW
# Reason: neither upstream repository provides a reusable ROS perception package.

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["rekpiper_perception"],
    package_dir={"": "src"},
)

setup(**setup_args)
