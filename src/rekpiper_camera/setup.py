#!/usr/bin/env python3

# Source classification: NEW
# Reason: neither ReKep repository provides a ROS D435 package.

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["rekpiper_camera"],
    package_dir={"": "src"},
)

setup(**setup_args)
