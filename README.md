# Internal wave libraries

 - Solve 1D KdV equation
 - Calculate linear dynamic modes
 - Fit dynamic modes to mooring data

---

# Installation

## Option 1 - pip install in develop mode:
 - Step 1) Clone this repository
 - Step 2) cd into repo
 - Step 3) pip install -e ./

## Option 2 - change environment variables [not really an install]
 - Step 1) Clone this repository
 - Step 2) Set PYTHONPATH environment variable to point to the path where the repo sits
 - Step 3) Install the dependencies below i.e. conda install ..., yum install ...

#### Dependencies
 - numpy
 - scipy
 - matplotlib
 - xarray
 - gsw (Gibbs seawater toolbox)

---

# Documentation

 - An example ipython notebook is in `tests` and viewed [here](https://nbviewer.jupyter.org/urls/bitbucket.org/mrayson/iwaves/raw/c42dd64008eb014049b80031c07eb832d372f57f/tests/test_kdvlamb.ipynb)
 - A draft paper describing the numerical scheme is [here](https://www.overleaf.com/read/wvvthjwgtxft)
---

Matt Rayson

University of Western Australia

August 2017
