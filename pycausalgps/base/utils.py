import random
import numpy as np
import pandas as pd


from pycausalgps.log import LOGGER


def nested_get(d, keys, default=None):
    """ Get a value from a nested dictionary.

    Parameters
    ----------
    d : dict
        The dictionary to search.
    keys : list
        The list of keys to search for.

    Returns
    -------
    value : object
        The value of the key if found, otherwise None.
    """
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

def generate_syn_pop(sample_size: int, seed_val:int, outcome_sd:int,
                             gps_spec:int, cova_spec:int) -> pd.DataFrame:
    """ Generate synthetic data

    Parameters
    ----------
    sample_size: int
        A number of required data samples.
    seed_val: int 
        A seed value for generating reproducible data.
    outcome_sd: int
        TBD
    gps_spec: int
        TBD
    cova_spec: int
        TBD

    Returns
    -------
    data: pd.DataFrame
        A dataframe containing the generated data.

    >>> md = generate_syn_pop(100)
    >>> len(md)
    100
    """
    
    random.seed(seed_val)

    mean = [0,0,0,0]
    cov = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    cf = np.random.multivariate_normal(mean, cov, sample_size).T

    cf5 = np.random.choice([-2,-1,0,1,2], size = sample_size, replace=True)
    cf6 = np.random.uniform(low=-3, high=3, size=sample_size)

    if  gps_spec == 1:
        treat = ((- 0.8 + 0.1 * cf[0] + 0.1 * cf[1] - 0.1 * cf[2] +
                    0.2 * cf[3] + 0.1 * cf5 + 0.1 * cf6) * 9 + 17  +
                    np.random.normal(loc=0.0, scale = 5, size=sample_size))

    elif gps_spec == 2:

        treat = ((- 0.8 + 0.1 * cf[0,:] + 0.1 * cf[1,:] - 0.1 * cf[2,:]
                  + 0.2 * cf[3,:] + 0.1 * cf5 + 0.1 * cf6) * 15 + 22 +
                   np.random.standard_t(size=sample_size,df=2))

        treat = pd.DataFrame({'t':treat})
        treat.loc[treat.t < -5, 't'] = -5
        treat.loc[treat.t > 25, 't'] = 25

        treat = treat['t'].tolist() 

    elif gps_spec == 3:
        
        treat = ((- 0.8 + 0.1 * cf[ 0, :] + 0.1 * cf[ 1, :]- 0.1 *cf[ 2,:] 
           + 0.2 * cf [3, :]
           + 0.1 * cf5 + 0.1 * cf6) * 9
           + 1.5 * pow(cf[ 2, :],2) 
           + np.random.normal(loc=0, scale = 5, size=sample_size) + 15)

    elif gps_spec == 4:
        
        treat = (49 * np.exp((-0.8 + 0.1 * cf[ 0,:] 
                + 0.1 * cf[ 1, :] - 0.1 * cf[ 2, :]
                + 0.2 * cf[ 3, :] + 0.1 * cf5 + 0.1 * cf6))
                / (1 + np.exp((-0.8 + 0.1 * cf[0,:] 
                + 0.1 * cf[ 1, :] - 0.1 * cf[ 3, :]
                + 0.2 * cf[ 3, :] + 0.1 * cf5 + 0.1 * cf6))) - 6 
                + np.random.normal(loc=0, scale=5, size=sample_size))
        
    elif gps_spec == 5:
        treat = (42 / (1 + np.exp((-0.8 + 0.1 * cf[ 0, :] 
                + 0.1 * cf[ 1, :]- 0.1 * cf[ 3, :]
                + 0.2 * cf[3,:] + 0.1 * cf5 + 0.1 * cf6))) - 18 
                + np.random.normal(loc=0, scale=5, size=sample_size))
    elif gps_spec == 6:
        treat = (np.log(abs(-0.8 + 0.1 * cf[ 0, :] + 0.1 * cf[ 1, :] 
                - 0.1 * cf[ 2, :] + 0.2 * cf[ 3, :] 
                + 0.1 * cf5 + 0.1 * cf6)) * 7 + 13 
                + np.random.normal(loc=0, scale=4, size=sample_size))
    elif gps_spec == 7:
        treat = ((-0.8 + 0.1 * cf[0,:] + 0.1 * cf[1,:] 
                 - 0.1 * cf[2,:] + 0.2 * cf[3,:]
                 + 0.1 * cf5 + 0.1 * cf6) * 15 + 22 
                 + np.random.standard_t(size=sample_size, df=2))
    else:
        raise ValueError(f"gps_spec:  {gps_spec} is not a valid value.")

    # produce outcome
    Y=np.zeros(sample_size)
    
    for i in range(sample_size):
        Y[i] = ((-(1 + (sum([0.2, 0.2, 0.3, -0.1] * cf[:,i]) * 10) - 
                  (2 * cf5[i]) - (2 * cf6[i]) +
                  (treat[i]-20) * (0.1 + 0.1 * cf[3,i] + 0.1 * cf5[i] +
                   0.1 * pow(cf[2,i],2) - pow(0.13,2) * pow(treat[i] - 20,2)))
                ) + np.random.normal(loc=0, scale=outcome_sd,size=1))[0]


    if cova_spec == 1:
        pass
    elif cova_spec == 2:
        cf[1,:] = np.exp(cf[0,:]/2)
        cf[2,:] = (cf[1,:]/(1+np.exp(cf[0,:])))+10
        cf[3,:] = (cf[0,:] * cf[2,:]/25 + 0.6) ^ 3
        cf[4,:] = (cf[1,:] + cf[3,:] + 20) ^ 2
    else:
        raise ValueError(f"cova_spec:  {cova_spec} is not a valid value.")

    
    data = pd.DataFrame({'Y':Y, 
                                   'treat':treat,
                                   'cf1':cf[0,:],
                                   'cf2':cf[1,:],
                                   'cf3':cf[2,:],
                                   'cf4':cf[3,:],
                                   'cf5':cf5,
                                   'cf6':cf6})

    return data



def human_readible_size(nbytes):
    """
    Convert a number of bytes to a human readable string. 
    """
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1

    return f"{nbytes:.2f} {suffixes[i]}"