#!/usr/bin/env python

from setuptools import setup #, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

		
setup(name='iwaves',
      version='latest',
      description='Internal wave KdV solver',
      author='Matt Rayson',
      author_email='matt.rayson@uwa.edu.au',
      #packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      packages=['iwaves.utils','iwaves.kdv'],
      install_requires=['numpy','scipy','matplotlib','gsw','netcdf4','xarray'],
      license='LICENSE',
      include_package_data=True,
      distclass=BinaryDistribution,
    )
