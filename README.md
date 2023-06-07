# pycausalgps

[![Test Status](https://github.com/nsaph-software/pycausalgps/workflows/Python%20package/badge.svg?branch=develop&event=push)](https://github.com/nsaph-software/pycausalgps/actions)
[![PyPI version](https://img.shields.io/pypi/v/pycausalgps.svg)](https://pypi.org/project/pycausalgps)
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
## Usage

### Geneting synthetic data

```python
from pycausalgps.base.utils import generate_syn_pop
data = generate_syn_pop(sample_size=1000, 
                        seed_val=456, 
                        outcome_sd=0.25, 
                        gps_spec=1, 
                        cova_spec=2)
```

### Estimating GPS

```python
from pycausalgps.base.utils import generate_syn_pop
from pycausalgps.base import GeneralizedPropensityScore

params = {"gps_density": "normal",
          "exposure_column": "treat",
          "covariate_column_num": ["cf1", 
                                   "cf2", 
                                   "cf3", 
                                   "cf4", 
                                   "cf6"],
          "covariate_column_cat": ["cf5"],
          "libs":{
                  "xgboost":{
                             "n_estimators": 100,
                             "max_depth": 3,
                             "learning_rate": 0.1,
                             "test_rate": 0.2,
                             "random_state": 42
                            }
                }
}

data = generate_syn_pop(sample_size=1000, 
                        seed_val=456, 
                        outcome_sd=0.25, 
                        gps_spec=1, 
                        cova_spec=2)

gps = GeneralizedPropensityScore(data, params)
```

### Generating Pseudo Population

- **Weighting approach**

```python
from pycausalgps.pseudo_population import PseudoPopulation


gps_params = {"gps_density": "normal",
              "exposure_column": "treat",
              "covariate_column_num": ["cf1", 
                                       "cf2", 
                                       "cf3", 
                                       "cf4", 
                                       "cf6"],
              "covariate_column_cat": ["cf5"],
              "libs":{
                      "xgboost":{
                             "n_estimators": 100,
                             "max_depth": 3,
                             "learning_rate": 0.1,
                             "test_rate": 0.2,
                             "random_state": 42
                             }
                         }
}

data = generate_syn_pop(sample_size=1000, 
                        seed_val=456, 
                        outcome_sd=0.25, 
                        gps_spec=1, 
                        cova_spec=2)
    
gps = GeneralizedPropensityScore(data, gps_params)

results = gps.get_results()
gps_data = {
    'data' : results.get("data"),
    'gps_density' : results.get("gps_density")
}



pspop_params = {"approach" : "weighting", 
                "exposure_column": "treat",
                "covariate_column_num": ["cf1", 
                                         "cf2", 
                                         "cf3", 
                                         "cf4", 
                                         "cf6"],
                "covariate_column_cat": ["cf5"],}

pspop = PseudoPopulation(data=data, 
                         gps_data=gps_data, 
                         params=pspop_params)

```

- **Matching approach**

```python
from pycausalgps.base.utils import generate_syn_pop
from pycausalgps.gps import GeneralizedPropensityScore
from pycausalgps.pseudo_population import PseudoPopulation


gps_params = {"gps_density": "normal",
              "exposure_column": "treat",
              "covariate_column_num": ["cf1", 
                                       "cf2", 
                                       "cf3", 
                                       "cf4", 
                                       "cf6"],
              "covariate_column_cat": ["cf5"],
              "libs":{
                      "xgboost":{
                             "n_estimators": 100,
                             "max_depth": 3,
                             "learning_rate": 0.1,
                             "test_rate": 0.2,
                             "random_state": 42
                             }
                         }
}

data = generate_syn_pop(sample_size=500, 
                        seed_val=456, 
                        outcome_sd=0.25, 
                        gps_spec=1, 
                        cova_spec=2)
    
gps = GeneralizedPropensityScore(data, gps_params)

results = gps.get_results()
gps_data = {
    'data' : results.get("data"),
    'gps_density' : results.get("gps_density"),
    'gps_minmax': results.get("gps_minmax")
}



pspop_params = {"approach" : "matching", 
                "exposure_column": "treat",
                "covariate_column_num": ["cf1", 
                                         "cf2", 
                                         "cf3", 
                                         "cf4", 
                                         "cf6"],
                "covariate_column_cat": ["cf5"],
                "control_params": {"caliper": 1.0,
                                   "scale": 0.5,
                                   "dist_measure": "l1",
                                   "bin_seq": None},
                "run_params": {"n_thread": 12,
                               "chunk_size": 500},
}

pspop = PseudoPopulation(data=data, 
                         gps_data=gps_data, 
                         params=pspop_params)
```

## Documentation

Documentation is hosted at https://nsaph-software.github.io/pycausalgps/
