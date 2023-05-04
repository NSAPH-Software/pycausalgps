"""
pseudo_population.py
====================
The core module for the PseudoPopulation class.
"""

import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from matplotlib.gridspec import GridSpec

from pycausalgps.log import LOGGER
from pycausalgps.base.utils import nested_get
from pycausalgps.rscripts.rfunctions import (compute_density,
                                             compute_absolute_weighted_corr)

class PseudoPopulation:
    """
    The PseudoPopulation class is used to compute the pseudo-population.

    Parameters
    ----------

    data: pd.DataFrame
        The dataframe containing the data and the GPS values as well as the
        estimated exposure and estimated exposure standard deviation.
    params: dict
        The dictionary containing the parameters for computing the 
        pseudo-population. The required parameters are:
        - approach: str
            The approach to compute the pseudo-population. The available
            approaches are: "weighting" and "matching".
        - exposure_column: str  
            A string that indicates the name of the exposure column.
        - covariate_column_num: list
            A list of strings that indicates the names of the numerical
            covariate columns.
        - covariate_column_cat: list
            A list of strings that indicates the names of the categorical
            covariate columns.

    """

    def __init__(self, data: pd.DataFrame, params: dict):

        self.data = data
        self.params = params
        self.compiling_report = {}
        self.results = None
        self._check_data()
        self._check_params()
        self.generate_pseudo_population()

    
    def _check_data(self) -> None:
        pass
    
    def _check_params(self):
        """
        Check if the params are in the correct format.
        """

        required_params = ["approach"]
        
        for param in required_params:
            if not param in self.params.keys():
                raise ValueError(f"Required parameter {param} is missing.")
            

    def generate_pseudo_population(self) -> None:
        """
        Compute the pseudo-population.
        """
        
        # For computing pspop we need to have the following:
        # 1. gps_params (gps values, estimated exposure, estimated exposure std,
        #                w_resid)
        # 2. pspop_params (approach, caliper, scale, 

        if self.params.get("approach") == "weighting":
            counter_weight = self._compute_pspop_weighting()
            self.counter_weight = counter_weight["counter_weight"]
            self._compute_covariate_balance()
        elif self.params.get("pspop_params").get("approach") == "matching":
            counter_weight = self._compute_pspop_matching()
            self.counter_weight = counter_weight["counter_weight"]
            self._compute_covariate_balance()
        else:
            raise Exception("Approach is not defined.")
    
    
    def _compute_pspop_weighting(self) -> pd.DataFrame:
        """
        Compute the pseudo-population using weighting.
        """
        
        # The weight of each sample is equal to the probablity of getting 
        # that exposure in the data over the probablity of getting that 
        # exposure given the covariates (GPS).

        # temp test
        compute_density_with_r = True
        
        exposure_data_col_name = nested_get(self.params, ["exposure_column"])

        # compute density of the exposure in the data.
        if compute_density_with_r:
            # compute density with R.
            w_val = self.data[exposure_data_col_name].to_numpy()
            data_density = compute_density(w_val, w_val)
            print("Density was computed with R.")
        else:
            exp_data = self.data[exposure_data_col_name]
            kde = gaussian_kde(exp_data)
            data_density = kde(exp_data)

        
        ipw = data_density / self.data["gps"].to_numpy()     
        value = pd.DataFrame({"id": self.data["id"], "counter_weight": ipw})

        return value
  

    def _compute_pspop_matching(self) -> None:

        """ 
        Compute the pseudo-population using matching approach.
        """

        # We need: 
        # Original GPS value and mean and standard deviation.
        # Original exposure value.
        # Requested exposure level. 
        # Caliper value.
        # Scale value.

        
        # load gps object from the database.
        gps_obj = self.db.get_value(self.gps_params.get("hash_value"))
        gps_min, gps_max = gps_obj._data.get("gps_minmax")
        gps_model = gps_obj.gps_params.get("gps_params").get("model")
        if gps_model == "non-parametric":
            w_resid = gps_obj.w_resid
     
        # load exposure values
        obs_exposure_data = self.load_exposure()
        w_min, w_max = (min(obs_exposure_data["exposure"]), 
                        max(obs_exposure_data["exposure"]))

        # controlling parameters
        delta = nested_get(self.pspop_params,
                               ["pspop_params", 
                                "controlling_params", 
                                "caliper"])
        scale = nested_get(self.pspop_params,
                                ["pspop_params",
                                 "controlling_params",
                                 "scale"])
        distance_metric = nested_get(self.pspop_params, 
                                          ["pspop_params",
                                           "controlling_params", 
                                           "distance_metric"])

        # collect requested exposure level.
        req_exposure = nested_get(self.pspop_params, 
                                       ["pspop_params", 
                                        "controlling_params", 
                                        "bin_seq"])
        
        # check if req_exposure is string
        if isinstance(req_exposure, str):
            req_exposure = eval(req_exposure)

        if req_exposure is None:
            req_exposure = np.arange(w_min+delta/2, w_max, delta)


        for i, w in enumerate(req_exposure):
            
            if gps_model == "parametric":
                p_w = norm.pdf(w, 
                               gps_obj._data.get("_data")["e_gps_pred"], 
                               gps_obj._data.get("_data")["e_gps_std_pred"])
            elif gps_model == "non-parametric":
                w_new = ((w - gps_obj._data.get("_data")["e_gps_pred"]) 
                         / gps_obj._data.get("_data")["e_gps_std_pred"])
                p_w = compute_density(w_resid, w_new)
            else:
                raise Exception("GPS model is not defined.")
           


            # select subset of data that are within the caliper value.
            subset_idx = np.where(np.abs(obs_exposure_data["exposure"] - w) <= delta)[0]
            subset_row = obs_exposure_data.iloc[subset_idx]["id"]

            # standardize GPS and Exposure values.
            std_w = (w - w_min)/(w_max - w_min)
            std_gps = (p_w - gps_min)/(gps_max - gps_min)

            # all data (observational data where w is requested w.)
            # std_w: scaler
            # std_gps: vector
            all_curated_data = pd.DataFrame({"id": obs_exposure_data["id"],
                                             "std_w": std_w,
                                             "std_gps": std_gps})
            
            # subset of data (actual standardized data that are within the caliper value.)
            # std_w_subset: vector
            # std_gps_subset: vector
            std_w_subset = ((obs_exposure_data[obs_exposure_data["id"].
                            isin(subset_row)]["exposure"] - w_min)) / (w_max - w_min)
            
            # TODO: use query.
            std_gps_subset = (gps_obj._data.get("_data")[gps_obj.
                              _data.get("_data")["id"].
                              isin(subset_row)]["gps_standardized"])

            # a: subset of data with standardized GPS (std_gps_subset)
            # b: all data with estimated GPS based on requested exposure level. (std_gps)
            # c: subset of data with standardized exposure (std_w_subset)
            # d: the exposure level that is requested. (std_w)
            
            a = std_gps_subset.to_numpy()
            b = std_gps
            c = std_w_subset.to_numpy()
            d = std_w

            c_minus_d = abs(c - d)*(1-scale)

            # compute closest sample from subset of data to these hypothetical samples.
            # TODO: make this a fucntion
            len_b = len(b)
            len_a = len(a)
            out = np.zeros(len_b)

            # counter_weight = pd.DataFrame.from_dict({key: 0 for key in 
            #                                          obs_exposure_data["id"]}, 
            #                                          orient="index", 
            #                                          columns=['counter_weight'])

            counter_weight = Counter({key: 0 for key in obs_exposure_data["id"]})
 
            # Brute force method.
            for i in range(len_b):
                for j in range(len_a):

                    subtract_val = abs(a[j] - b[i])*scale

                    tmp_val = subtract_val + c_minus_d[j]

                    if (j == 0):
                        min_val = tmp_val
                        min_idx = j
                        continue

                    if (tmp_val < min_val):
                        min_val = tmp_val
                        min_idx = j
                out[i] = min_idx
            
            # Solution with numpy, but it will consume more memory.
            # Keeping this for future reference.
            out2 = np.argmin(np.abs(np.subtract.outer(a, b))*scale 
                            + c_minus_d[:, np.newaxis], axis=0)

            

            # Get the id based on the index.
            selected_id = subset_row.iloc[out].to_numpy()
            tmp_freq_table = Counter(selected_id)
            counter_weight.update(tmp_freq_table)   

        counter_weight_dict = dict(counter_weight)
        counter_weight_df = pd.DataFrame(list(counter_weight_dict.items()), 
                                                   columns=["id", "counter_weight"])

        return counter_weight_df

            
    def _compute_covariate_balance(self) -> None:
        """
        Compute covariate balance.
        """
        if self.counter_weight is None:
            raise Exception("Counter weight is not defined. " +\
                "Try generating the pseudo-population first.")

        # load exposure values.
        # exposure_data = self.load_exposure()
        # exp_data = exposure_data[self.gps_params.get("gps_params").get("pred_model").get("exposure_column")].to_numpy()
        exposure_data_col_name = nested_get(self.params, ["exposure_column"])
        exp_data = self.data[exposure_data_col_name].to_numpy()

        # load the confounders.
        covariate_col_num = nested_get(self.params, 
                                       ["covariate_column_num"])
        covariate_col_cat = nested_get(self.params, 
                                       ["covariate_column_cat"])
        covariate_data_num = self.data[covariate_col_num]
        covariate_data_cat = self.data[covariate_col_cat]

        # compute weighted correlation between the confounders and the exposure.
        covariate_balance = compute_absolute_weighted_corr(exp_data, 
                                                           self.counter_weight, 
                                                           covariate_data_num,
                                                           covariate_data_cat) 

        original_data_covariate_balance = compute_absolute_weighted_corr(
            exp_data,
            np.ones(self.counter_weight.shape),
            covariate_data_num,
            covariate_data_cat)                    

        cov_balance_current = covariate_balance
        cov_balance_current.rename(columns = {'value':'current'}, 
                                   inplace = True)
        cov_balance_original = original_data_covariate_balance
        cov_balance_original.rename(columns = {'value':'original'}, 
                                    inplace = True)


        cov_balance = pd.merge(cov_balance_current, cov_balance_original, 
                               on='name')

        self.covariate_balance = cov_balance


    def plot_cov_balance(self) -> None:
        """
        Plot the covariate balance.
        """
        if self.covariate_balance is None:
            raise Exception("Covariate balance is not defined. " +\
                "Try generating the pseudo-population first.")

        cov_data = self.covariate_balance
        cov_data.sort_values(by=['original'], inplace=True)
        cov_data.reset_index(drop=True, inplace=True)
        flipped_cov_data = cov_data.iloc[::-1]
        flipped_cov_data.reset_index(drop=True, inplace=True)

        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        fig.tight_layout()
        gs = GridSpec(3, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[0:2, -1])
        ax3 = fig.add_subplot(gs[2, :])

        org_color = "#DC3220"
        adj_color = "#005AB5"

        # ax 1: main plot ------------------------------------------------------
        ax1.plot(cov_data['original'],cov_data['name'], '-o', color = org_color)
        ax1.plot(cov_data['current'],cov_data['name'], '-o', color = adj_color)
        ax1.set_xlabel('Absolute Correlation')
        ax1.set_ylabel('Covariates')
        ax1.grid(True)
        ax1.legend(['Original','Adjusted'])

        # ax 2: plot info ------------------------------------------------------
        ax2.axis('off')
        xlim2 = ax2.get_xlim()
        ylim2 = ax2.get_ylim()

        hyper_param_1 = {"fontweight": "bold", "fontsize": 8, "ha":"left",
                         "va":"center"}

        ax2.text(xlim2[0], ylim2[1]-0.01, 'Covariates', **hyper_param_1)
        ax2.text(xlim2[0]+0.4, ylim2[1]-0.01, 'Original', **hyper_param_1)
        ax2.text(xlim2[0]+0.7, ylim2[1]-0.01, 'Adjusted', **hyper_param_1)
 
        
        for i in range(flipped_cov_data.shape[0]):
            if len(flipped_cov_data.loc[i,'name']) > 8:
                mstr = flipped_cov_data.loc[i,'name'][0:3] + '*' +\
                flipped_cov_data.loc[i,'name'][-3:]
            else:
                mstr = flipped_cov_data.loc[i,'name']
            ax2.text(0.02, (1-(i+2)/25)*1, mstr, ha='left', va='center',
                     fontsize=6, fontstyle='italic')

        hyper_param_2 = {"fontsize": 7, "ha":"left", "va":"center",
                         "fontstyle": "italic"}
        for i in range(flipped_cov_data.shape[0]):
            ax2.text(xlim2[0]+0.4, (1-(i+2)/25)*1, 
            f"{flipped_cov_data.loc[i,'original']:0.3f}", color = org_color,
             **hyper_param_2)
            ax2.text(xlim2[0]+0.7, (1-(i+2)/25)*1, 
            f"{flipped_cov_data.loc[i,'current']:0.3f}", color = adj_color,
             **hyper_param_2)

        # ax 3: Data info ------------------------------------------------------
        ax3.axis('off')
        xlim3 = ax3.get_xlim()
        ylim3 = ax3.get_ylim()

        hyper_param_3 = {"fontsize": 8, "ha":"left", "va":"center"}

        ax3.text(xlim3[0], ylim3[1]-0.05, 'Data Info', **hyper_param_1)
        # ax3.text(xlim3[0], ylim3[1]-0.15, 
        #          f'GPS Object ID: {self.gps_params.get("gps_id")}', 
        #          **hyper_param_3)
        # ax3.text(xlim3[0], ylim3[1]-0.2,
        #             f'GPS hash value: {self.gps_params.get("hash_value")}',
        #             **hyper_param_3)
        # ax3.text(xlim3[0], ylim3[1]-0.25,
        #             f'Pseudo population ID: {self.pspop_id}',
        #             **hyper_param_3)
        # ax3.text(xlim3[0], ylim3[1]-0.3,
        #             f'Pseudo population hash value: {self.hash_value}',
        #             **hyper_param_3)

        fig.suptitle("Covariate Balance", fontsize=16)
        plt.show()   
    
    
    
    
    
    # Old code 
    
    

if __name__ == "__main__":

    from pycausalgps.base.utils import generate_syn_pop
    from pycausalgps.gps import GeneralizedPropensityScore

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
    gps_data = results.get("data")    

    merged_df = pd.merge(data, gps_data, on='id')
    merged_df

    pspop_params = {"approach" : "weighting", 
                    "exposure_column": "treat",
                    "covariate_column_num": ["cf1", 
                                             "cf2", 
                                             "cf3", 
                                             "cf4", 
                                             "cf6"],
                    "covariate_column_cat": ["cf5"]}
    

