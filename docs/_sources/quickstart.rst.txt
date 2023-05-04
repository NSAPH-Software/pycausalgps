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
                                 "random_state": 42
                                 }
                             }
    }
    
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