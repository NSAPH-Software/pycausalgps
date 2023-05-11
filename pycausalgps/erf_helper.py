from scipy.stats import norm
import numpy as np
import pandas as pd
from pycausalgps.log import LOGGER
from pycausalgps.rscripts.rfunctions import locpol
from multiprocessing import Pool


def generate_kernel(t):
    return norm.pdf(t)

def w_fun(bw, matched_w, w_vals):
    w_avals = []
    for w_val in w_vals:
        w_std = (matched_w - w_val) / bw
        kern_std = generate_kernel(w_std) / bw
        tmp_mean = np.mean(w_std ** 2 * kern_std)
        w_avals.append(tmp_mean * (generate_kernel(0) / bw) /
                       (np.mean(kern_std) * tmp_mean - 
                        np.mean(w_std * kern_std) ** 2))
    return np.array(w_avals) / len(matched_w)

def estimate_hat_vals(bw, matched_w, w_vals):
    return np.interp(matched_w, w_vals, w_fun(bw, matched_w, w_vals), right=0)

def smooth_erf(matched_Y, bw, matched_w, matched_cw):

    if sum(matched_cw == 0) == len(matched_cw):
        matched_cw = matched_cw + 1
        LOGGER.debug("Giving equal weight for all samples.")

    data = pd.DataFrame({'m_Y': matched_Y, 
                         'm_w': matched_w, 
                         'counter_weight': matched_cw})
    smoothed_val = locpol(data, 'm_Y ~ m_w', bw, matched_w)

    return smoothed_val


def compute_risk(h, matched_Y, matched_w, matched_cw, w_vals):
    hats = estimate_hat_vals(h, matched_w, w_vals)
    tmp_mean = np.mean(((matched_Y - smooth_erf(
                         matched_Y,
                         bw = h,
                         matched_w = matched_w,
                         matched_cw = matched_cw)) / (1 - hats)) ** 2)
    return tmp_mean

def compute_risk_par_helper(args):
    return compute_risk(*args)


def estimate_npmetric_erf(m_Y, m_w, counter_weight, bw_seq, w_vals, nthread):
    if len(m_Y) != len(m_w):
        raise ValueError("Length of output and treatment should be equal!")

    if not (isinstance(m_Y, np.ndarray) and isinstance(m_w, np.ndarray)):
        raise ValueError("Output and treatment vectors should be numpy arrays.")

    if sum(counter_weight == 0) == len(counter_weight):
        counter_weight = counter_weight + 1
        LOGGER.debug("Giving equal weight for all samples.")
    
    args = [(bw, m_Y, m_w, counter_weight, w_vals) for bw in bw_seq]
    with Pool(nthread) as pool:
        risk_val = pool.map(compute_risk_par_helper, args)

    risk_val = np.array(risk_val)


    h_opt = bw_seq[np.argmin(risk_val)]

    LOGGER.info(f"The band width with the minimum risk value: {h_opt}.")

    tmp_data = pd.DataFrame({'m_Y': m_Y, 
                             'm_w': m_w, 
                             'counter_weight': counter_weight})
    
    erf = locpol(tmp_data, 'm_Y ~ m_w', h_opt, w_vals)

    if sum(np.isnan(erf)) > 0:
        LOGGER.debug(f"erf has {sum(np.isnan(erf))} missing values.")

    result = {
        'm_Y': m_Y,
        'm_w': m_w,
        'bw_seq': bw_seq,
        'w_vals': w_vals,
        'risk_val': risk_val,
        'h_opt': h_opt,
        'erf': erf,
    }

    return result




if __name__ == "__main__":
    import pandas as pd

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
    bw_seq = np.arange(0.1, 4.0, 0.05)
    w_vals = np.arange(5, 15, 1)
    nthread = 1

    erf_pmetric = estimate_npmetric_erf(m_Y, m_w, counter_weight, 
                                        bw_seq, w_vals, nthread)