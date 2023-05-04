"""
pseudo_population.py
====================
The core module for the PseudoPopulation class.
"""

import json
import hashlib
from collections import Counter
from dataclasses import dataclass

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
        The dataframe containing the data (covariate and exposure columns).
    gps_data: dict
        The dictionary containing the GPS data. The required keys are:
        - gps_data: pd.DataFrame
            The dataframe containing the GPS data, and its auxiliary columns.
        - gps_model: str
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

    APPROACH_WEIGHTING = "weighting"
    APPROACH_MATCHING = "matching"
    GPS_DENSITY_NORMAL = "normal"
    GPS_DENSITY_KERNEL = "kernel"

    def __init__(self, data: pd.DataFrame, gps_data: dict, params: dict):

        self.data = data
        self.params = params
        self.gps_data = gps_data
        self.compiling_report = {}
        self.counter_weight = None
        self.generate_pseudo_population()

    @property
    def data(self) -> pd.DataFrame:
        return self.__data
    
    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        self.__data = value

    @property
    def params(self) -> dict:
        return self.__params
    
    @params.setter
    def params(self, value: dict) -> None:
        #TODO: Check if the params are in the correct format.
        if not isinstance(value, dict):
            raise ValueError("Params must be a dictionary.")
        self.__params = value

    @property
    def gps_data(self) -> dict:
        return self.__gps_data
    
    @gps_data.setter
    def gps_data(self, value: dict) -> None:
        #TODO: check if the gps_data is in the correct format.
        if not isinstance(value, dict):
            raise ValueError("Data must be a pandas DataFrame.")
        self.__gps_data = value

    def generate_pseudo_population(self) -> None:
        """
        Compute the pseudo-population.
        """
        self.exposure_data_col_name = self.params.get("exposure_column")
        self.covariate_col_num = self.params.get("covariate_column_num")
        self.covariate_col_cat = self.params.get("covariate_column_cat")
        
        if self.params.get("approach") == self.APPROACH_WEIGHTING:
            self.counter_weight = self._compute_pspop_weighting() 
            self._compute_covariate_balance()
        elif self.params.get("approach") == self.APPROACH_MATCHING:
            self.counter_weight, self.counter_weight_list = self._compute_pspop_matching()
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
        
        
        # compute density of the exposure in the data.
        if compute_density_with_r:
            # compute density with R.
            w_val = self.data[self.exposure_data_col_name].to_numpy()
            data_density = compute_density(w_val, w_val)
            print("Density was computed with R.")
        else:
            exp_data = self.data[self.exposure_data_col_name]
            kde = gaussian_kde(exp_data)
            data_density = kde(exp_data)

        
        ipw = data_density / self.gps_data.get("data")["gps"].to_numpy()            
        value = pd.DataFrame({"id": self.gps_data.get("data")["id"], 
                              "counter_weight": ipw})

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
        gps_min, gps_max = self.gps_data.get("gps_minmax")
        gps_density = self.gps_data.get("gps_density")
        if gps_density == self.GPS_DENSITY_KERNEL:
            w_resid = self.gps_data.get("data")["w_resid"]
     
        # load exposure values
        obs_exposure_data = self.data[["id", self.exposure_data_col_name]]
        w_min, w_max = (min(obs_exposure_data[self.exposure_data_col_name]), 
                        max(obs_exposure_data[self.exposure_data_col_name]))

        # controlling parameters
        delta = nested_get(self.params,
                                ["control_params", 
                                "caliper"])
        scale = nested_get(self.params,
                                ["control_params",
                                "scale"])
        
        dist_measure = nested_get(self.params, 
                                 ["control_params", 
                                  "dist_measure"])

        # collect requested exposure level.
        req_exposure = nested_get(self.params, 
                                 ["control_params", 
                                  "bin_seq"])
        
        # check if req_exposure is string
        if isinstance(req_exposure, str):
            req_exposure = eval(req_exposure)

        if req_exposure is None:
            req_exposure = np.arange(w_min+delta/2, w_max, delta)
     
        counter_weight = Counter({key: 0 for key in obs_exposure_data["id"]})
        counter_weight_list = []

        for i, w in enumerate(req_exposure):
            if gps_density == "normal":
                p_w = norm.pdf(w, 
                               self.gps_data.get("data")["e_gps_pred"], 
                               self.gps_data.get("data")["e_gps_std_pred"])
            elif gps_density == "kernel":
                w_new = ((w - self.gps_data.get("data")["e_gps_pred"]) 
                         / self.gps_data.get("data")["e_gps_std_pred"])
                p_w = compute_density(w_resid, w_new)
            else:
                raise Exception("GPS model is not defined.")
           

            # select subset of data that are within the caliper value.
            subset_idx = np.where(np.abs(obs_exposure_data[self.exposure_data_col_name] - w) <= delta)[0]
            subset_row = obs_exposure_data.iloc[subset_idx]["id"]

            print(f"Subset size: {len(subset_row)}")

            if len(subset_row) == 0:
                print(f"No data within the caliper value for the requested exposure level: {w}.")
                continue 


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
                            isin(subset_row)][self.exposure_data_col_name] - w_min)) / (w_max - w_min)
            
            # TODO: use query.
            std_gps_subset = (self.gps_data.get("data")[self.gps_data.get("data")["id"].isin(subset_row)]["gps_standardized"])

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
            out2 = np.zeros(len_b)
            out3 = np.zeros(len_b)

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

            
            # Third solution:
            # Vectorized method
            for i in range(len_b):
                tmp_vals = np.abs(a - b[i]) * scale + c_minus_d
                min_idx = np.argmin(tmp_vals)
                out3[i] = min_idx

            print(f"out - out2: {sum(out - out2)}")
            print(f"out2 - out3: {sum(out2 - out3)}")




            # Get the id based on the index.
            selected_id = subset_row.iloc[out].to_numpy()
            tmp_freq_table = Counter(selected_id)
            #counter_weight.update(tmp_freq_table)   
            counter_weight_list.append(CounterWeightData(w, tmp_freq_table))


        for i_counter_weight in counter_weight_list:
            counter_weight.update(i_counter_weight.counter_weight)
        
        counter_weight_dict = dict(counter_weight)
        counter_weight_df = pd.DataFrame(list(counter_weight_dict.items()), 
                                                   columns=["id", "counter_weight"])

        return counter_weight_df, counter_weight_list

            
    def _compute_covariate_balance(self) -> None:
        """
        Compute covariate balance.
        """
        if self.counter_weight is None:
            raise Exception(f"Counter weight is not defined. " 
                             "Try generating the pseudo-population first.")

        
        # merge the counter weight with the original data.
        # TODO: warn users if there is a discrepancy in ids.
        self.merged_data = pd.merge(self.data, self.counter_weight, on="id")
               
        # load the exposure data.
        exp_data = self.merged_data[self.exposure_data_col_name].to_numpy()

        # load the confounders.
        covariate_data_num = self.merged_data[self.covariate_col_num]
        covariate_data_cat = self.merged_data[self.covariate_col_cat]
        counter_weight = self.merged_data["counter_weight"].to_numpy()

        # compute weighted correlation between the confounders and the exposure.
        covariate_balance = compute_absolute_weighted_corr(exp_data, 
                                                           counter_weight, 
                                                           covariate_data_num,
                                                           covariate_data_cat) 

        original_data_covariate_balance = compute_absolute_weighted_corr(
            exp_data,
            np.ones(counter_weight.shape),
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


    def get_results(self) -> dict:
        """
        Get the results of generating the pseudo-population.
        """

        results = {"data": self.merged_data,
                   "params": self.params,
                   "compiling_report": self.compiling_report,
                   "covariate_balance": self.covariate_balance}
        
        return results

    def get_individual_counter_weight(self) -> list:
        """
        Get the counter-weight for each individual exposure.
        """

        if self.counter_weight_list is None:
            raise Exception("Counter-weight is not defined. " +\
                "Try generating the pseudo-population first.")

        return self.counter_weight_list


@dataclass    
class CounterWeightData:
    w: float
    counter_weight: np.ndarray

    
    
    
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
    

