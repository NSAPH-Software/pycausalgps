Estimating Generalized Propensity Score (GPS)
=============================================

The conditional probability of assignment to a particular treatment given a vector 
of observed covariates is called propensity score (Rosenbaum and Rubin 1983). 
The extended version of the propensity score is called the generalized propensity 
score (Hirano and Imbens 2004). Users can compute a generalized propensity 
score (GPS) using the `ComputeGPS` class. There are two models implemented to 
estimate GPS (`gps_models`):

Parametric
---------- 

A parametric model is standard linear regression model. 

TODO: mathematical equation

.. math::

   GPS 

Non-parametric
--------------

A non-parametric model refers to a flexible machine-learning model. 

TODO: mathematical equation

.. math::

   GPS 


Computing GPS Values
====================

The ComputeGPS class can be used to generate GPS values. The following represents
a simple example.

.. code-block:: python

    from pycausalgps.gps_utils import generate_syn_pop
    from pycausalgps.gps import ComputeGPS
    
    data = generate_syn_pop(1000)
    data['cf5'] = data.cf5.astype('category')
    
    params = {'approach':"xgboost1", 'gps_model': 'parametric', 
              'test_size': 0.2, 'random_state': 1,
              'n_estimator': 1000, 'learning_rate': 0.01}
    
    conf = data[["cf1","cf2","cf3","cf4","cf5","cf6"]]
    treat = data[["treat"]]
    
    gp = ComputeGPS(X=conf, y=treat, params=params)
    
    gp.compute_gps()
    
    print(gp.gps)




