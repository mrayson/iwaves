#!/usr/bin/env python

from setuptools import setup #, find_packages

setup(name='iwaves',
      version='latest',
      description='Internal wave KdV solver',
      author='Matt Rayson',
      author_email='matt.rayson@uwa.edu.au',
      #packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
	  packages=['iwaves.utils','iwaves.kdv'],
      license='LICENSE',
    )