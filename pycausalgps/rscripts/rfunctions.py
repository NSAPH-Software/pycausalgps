"""
rfunctions.py
=============
Main module to define the R functions to be used in the package.
"""


import numpy as np
import pandas as pd

from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter



#### Compute density function --------------------------------------------------
# R 
# Check if function is defined
# TODO: Check if there is a better way to fix the path.
if 'compute_density' not in robjects.globalenv:
    try:
        robjects.r('''
            source('pycausalgps/rscripts/compute_density.r')
        ''')
    except Exception as e:
        print('Error in loading the R function: compute_density')
        print(e)



# Python 
# Kernel density estimation

def compute_density(x0:np.array, x1:np.array) -> np.array:
    """ Kernel density estimation. 
    params: 

    | x0: A vectore of values to compute the density function.    
    | x1: A vector of values to evaluate the density function.     

    returns:   

    | A vector of density values the same size as x1.    
    """
    
  
    # check input data
    if not isinstance(x0, robjects.vectors.FloatVector):
        x0_r = robjects.FloatVector(x0)
    if not isinstance(x1, robjects.vectors.FloatVector):
        x1_r = robjects.FloatVector(x1)
    
    # collect the function from R
    r_f = robjects.globalenv['compute_density']
    
    # call the function
    results = r_f(x0_r, x1_r)

    # return back the results into a numpy array
    np_results = np.array(results)
    
    
    # Checks
    if np_results.shape[0] != x1.shape[0]:
        raise ValueError('The output shape is not the same as the input shape.')

    return np_results


#### ---------------------------------------------------------------------------

#### Compute absolute weighted corr function -----------------------------------

# R
# Check if function is defined
if 'absolute_weighted_corr' not in robjects.globalenv:
    try:
        robjects.r('''
            source('pycausalgps/rscripts/absolute_weighted_corr.r')
        ''')
    except Exception as e:
        print('Error in loading the R function: absolute_weighted_corr')
        print(e)

# Python

def compute_absolute_weighted_corr(w:np.array, 
                                   vw:np.array,
                                   c_num:pd.DataFrame,
                                   c_cat:pd.DataFrame) -> pd.DataFrame:

    """ Compute the absolute weighted correlation using R's wCorr package.    
    
    params:    

    | w: A vector of exposure values.      
    | vw: A vector of weights.     
    | c_num: A dataframe of numerical covariates.    
    | c_cat: A dataframe of categorical covariates.    

    returns:
    
    | A dataframe of absolute weighted correlation values.
    """

    # check input data
    if not isinstance(w, robjects.vectors.FloatVector):
        w_r = robjects.FloatVector(w)
    if not isinstance(vw, robjects.vectors.FloatVector):
        vw_r = robjects.FloatVector(vw)
    
    #TODO: check if one of them is None.
    # convert pandas dataframe to R dataframe
    with localconverter(robjects.default_converter + pandas2ri.converter):
        c_num_r = robjects.conversion.py2rpy(c_num)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        c_cat_r = robjects.conversion.py2rpy(c_cat)
    
    # collect the function from R
    r_absolute_weighted_corr = robjects.globalenv['absolute_weighted_corr_df']

    # call the function
    results = r_absolute_weighted_corr(w_r, vw_r, c_num_r, c_cat_r)

    # return back the results into a pandas dataframe
    with localconverter(robjects.default_converter + robjects.pandas2ri.converter):
        results_python = robjects.conversion.rpy2py(results)

    return results_python


#### ---------------------------------------------------------------------------



if __name__ == "__main__":
    x0 = np.random.normal(0, 1, 100)
    x1 = np.random.normal(0, 1, 100)
    dnsty = compute_density(x0, x1)
    #print(dnsty)

    # plot x0 and dnsty 
    import matplotlib.pyplot as plt
    plt.plot(x0, dnsty, 'o')
    plt.show()