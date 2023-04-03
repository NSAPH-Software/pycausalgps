# pycausalgps

_[This packages is under development. Some of the features are not yet implemented.]_

[![Test Status](https://github.com/nsaph-software/pycausalgps/workflows/Python%20package/badge.svg?branch=develop&event=push)](https://github.com/nsaph-software/pycausalgps/actions)
[![PyPI version](https://badge.fury.io/py/pycausalgps.svg)](https://badge.fury.io/py/pycausalgps)
[![codecov](https://codecov.io/gh/NSAPH-Software/pycausalgps/branch/develop/graph/badge.svg)](https://codecov.io/gh/NSAPH-Software/pycausalgps)


Causal Inference with Generalized Propensity Score

pycausalgps is a Python library for implementing matching on generalized propensity scores with continuous exposures. We have developed an innovative approach for estimating causal effects using observational data in settings with continuous exposures and introduced a new framework for GPS caliper matching that jointly matches on both the estimated GPS and exposure levels to fully adjust for confounding bias.


## Setting Up Environment

Please refer to the documentation for setting up the enviorment.

## Installation

To install the latest release from PyPI, run:

```bash
pip install pycausalgps
```

To install the latest development version from GitHub, run:

```bash
git clone https://github.com/NSAPH-Software/pycausalgps
cd pycausalgps
pip install .
```

## Documentation

Documentation is hosted at https://nsaph-software.github.io/pycausalgps/
