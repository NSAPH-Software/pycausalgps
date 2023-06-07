Quickstart
==========

Installing the Package
----------------------

Install the package using pip:

.. code-block:: bash

    $ pip install pycausalgps

See the `Setting Up Environment <installation.html>`_ for more details.


Generating Synthetic Data
-------------------------

The package provides a function to generate synthetic data. 

.. code-block:: python

    >>> from pycausalgps.base.utils import generate_syn_pop
    >>> data = generate_syn_pop(sample_size=1000, 
                                seed_val=456, 
                                outcome_sd=0.25, 
                                gps_spec=1, 
                                cova_spec=2)



Estimating GPS values
---------------------

Two approaches are provided to estimate GPS values. These approaches are different 
based on the type of gps_denisty specified. These approaches are `normal` and `kernel`.

Example of `normal` approach:
+++++++++++++++++++++++++++++

.. code-block:: python

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


Example of `kernel` approach:
+++++++++++++++++++++++++++++


.. code-block:: python
    
    from pycausalgps.base.utils import generate_syn_pop
    from pycausalgps.base import GeneralizedPropensityScore

    params = {"gps_density": "kernel",
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


Generating Pseudo-population
----------------------------

There are two implemented methdos to generate pseudo-population. These methods
are: `weighting` and `matching`.

Example of `weighting` approach:
++++++++++++++++++++++++++++++++


.. code-block:: python

    data = generate_syn_pop(sample_size=1000, 
                            seed_val=456, 
                            outcome_sd=0.25, 
                            gps_spec=1, 
                            cova_spec=2)
    
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
                                 "random_state": 42,
                                 "n_jobs": 12
                                 }
                             }
    }
    
    gps = GeneralizedPropensityScore(data, gps_params)
    
    results = gps.get_results()
    gps_data = {
        'data' : results.get("data"),
        'gps_density' : results.get("gps_density")
        'gps_minmax': results.get("gps_minmax")
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


Example of `matching` approach:
+++++++++++++++++++++++++++++++

.. code-block:: python

    from pycausalgps.base.utils import generate_syn_pop
    from pycausalgps.gps import GeneralizedPropensityScore
    from pycausalgps.pseudo_population import PseudoPopulation
    
    
    # matching
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
                                 "random_state": 42,
                                 "n_jobs": 12
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


.. note::

   Note that the matching approach is computationally expensive and internally
   uses multiprocessing. If you encounter any runtime error, try put the code
   inside a `if __name__ == '__main__':` block.


Exposure Response Function
--------------------------

The package provides a function to estimate exposure response function. There
are three implemented methods to estimate exposure response function. These
methods are: `parametric`, `semiparametric` and `nonparametric`.

Example of `parametric` approach:
+++++++++++++++++++++++++++++++++

The parametric approach estimates the hazard ratios using a parametric
regression model. In the backend we us gnm R package to implement generalized 
additive models.

.. code-block:: python

    import pandas as pd

    from pycausalgps.base.utils import generate_syn_pop
    from pycausalgps.gps import GeneralizedPropensityScore
    from pycausalgps.pseudo_population import PseudoPopulation
    from pycausalgps.rscripts.rfunctions import estimate_pmetric_erf

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

    data = generate_syn_pop(sample_size=200, 
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

    ps_pop = PseudoPopulation(data=data, 
                              gps_data=gps_data, 
                              params=pspop_params)

    ps_results = ps_pop.get_results()
    erf_pmetric = estimate_pmetric_erf(formula="Y ~ treat", 
                                       family="gaussian", 
                                       data = ps_results.get("data"))

Example of `semiparametric` approach:
+++++++++++++++++++++++++++++++++++++

The semiparametric approach estimates the smoothed exposure-response function
using generalized additive model with splines. In the backend we us `gam` R 
package to implement generalized additive models.


.. code-block:: python

    # Code from pmetric approach
    from pycausalgps.rscripts.rfunctions import estimate_semipmetric_erf

    erf_pmetric = estimate_semipmetric_erf(formula="Y ~ treat", 
                                           family="gaussian", 
                                           data = ps_results.get("data"))


Example of `nonparametric` approach:
++++++++++++++++++++++++++++++++++++

The nonparametric approach estimates the smoothed exposure-response function
using kernel smoothing approach. In the backend we us `locpol` R package to 
implemetn locpol polynomial fitting with a kernel weight. A data-driven 
bandwidth selection is implemented. 


.. code-block:: python
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from pycausalgps.base.utils import generate_syn_pop
    from pycausalgps.gps import GeneralizedPropensityScore
    from pycausalgps.pseudo_population import PseudoPopulation
    from pycausalgps.erf_helper import estimate_npmetric_erf

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

    ps_pop = PseudoPopulation(data=data, gps_data=gps_data, params=pspop_params)
    ps_results = ps_pop.get_results()

    m_Y = ps_results.get("data")["Y"].to_numpy()
    m_w = ps_results.get("data")["treat"].to_numpy()
    counter_weight = ps_results.get("data")["counter_weight"].to_numpy()
    bw_seq = np.arange(0.1, 4.0, 0.01)
    w_vals = np.arange(1, 20, 0.2)
    nthread = 12

    erf_pmetric = estimate_npmetric_erf(m_Y, m_w, counter_weight, 
                                    bw_seq, w_vals, nthread)


