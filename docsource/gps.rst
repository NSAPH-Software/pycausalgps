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




