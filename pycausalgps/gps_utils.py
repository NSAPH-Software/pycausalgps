import random
import numpy as np
import pandas as pd

from .dataset import Dataset


def gen_synthetic_population(sample_size, seed_val=300, outcome_sd=10,
                             gps_spec=1, cova_spec=1):
    

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

    
    simulated_data = pd.DataFrame({'Y':Y, 'treat':treat,
                                   'cf1':cf[0,:],
                                   'cf2':cf[1,:],
                                   'cf3':cf[2,:],
                                   'cf4':cf[3,:],
                                   'cf5':cf5,
                                   'cf6':cf6})

    return Dataset(simulated_data)

if __name__ == "__main__":
    sim_data = gen_synthetic_population(1000, gps_spec=1)
    print(sim_data)