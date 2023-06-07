"""
pseudo_population.py
====================
The core module for the PseudoPopulation class.
"""

from collections import Counter
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
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
            raise ValueError("GPS data must be a dictionary.")
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
            (
             self.counter_weight, 
             self.counter_weight_list
            ) = self._compute_pspop_matching()
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
  
    @staticmethod
    def compute_min_idx(i, a, b, scale, c_minus_d):
        tmp_vals = np.abs(a - b[i]) * scale + c_minus_d
        min_idx = np.argmin(tmp_vals)
        return min_idx

    @staticmethod
    def compute_min_idx_proc(args):
        i, a, b, scale, c_minus_d = args
        tmp_vals = np.abs(a - b[i]) * scale + c_minus_d
        min_idx = np.argmin(tmp_vals)
        return min_idx
    
    @staticmethod
    def compute_min_idx_proc_chunk(args):
        start_idx, chunk_size, a, b, scale, c_minus_d = args
        out_chunk = np.full(chunk_size, -1,  dtype=np.int)  # Initialize with np.nan
        for i in range(chunk_size):
            idx = start_idx + i
            if idx >= len(b):
                break
            tmp_vals = np.abs(a - b[idx]) * scale + c_minus_d
            min_idx = np.argmin(tmp_vals)
            out_chunk[i] = min_idx
        return out_chunk


    def _process_exposure_level(self, w):

        # load gps object from gps_data
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

        
        if gps_density == self.GPS_DENSITY_NORMAL:
            p_w = norm.pdf(w, 
                           self.gps_data.get("data")["e_gps_pred"], 
                           self.gps_data.get("data")["e_gps_std_pred"])
        elif gps_density == self.GPS_DENSITY_KERNEL:
            w_new = ((w - self.gps_data.get("data")["e_gps_pred"]) 
                      / self.gps_data.get("data")["e_gps_std_pred"])
            p_w = compute_density(w_resid, w_new)
        else:
            raise Exception("GPS model is not defined.")
           

        # select subset of data that are within the caliper value.
        subset_idx = np.where(np.abs(
            obs_exposure_data[self.exposure_data_col_name] - w) <= delta)[0]
        subset_row = obs_exposure_data.iloc[subset_idx]["id"]

        if len(subset_row) == 0:
            LOGGER.debug(f"No data found within the caliper value ({delta}) " 
                         f"for the requested exposure level: {w}.")
            return (w, Counter(None))


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
        #len_a = len(a)
        out = np.zeros(len_b)

        # chunk_size = int(nested_get(self.params, ["run_params", "chunk_size"]))
        # num_chunks = int(np.ceil(len_b / chunk_size))

        # with ProcessPoolExecutor(max_workers=n_thread) as executor:
        #     args_list = [(i * chunk_size, chunk_size, a, b, scale, c_minus_d) for i in range(num_chunks)]
        #     results = list(executor.map(self.compute_min_idx_proc_chunk, args_list))

        # # Flatten the results and assign them back to the 'out' array
        # out = np.concatenate([chunk[chunk != -1] for chunk in results])

    
        for i in range(len_b):
            tmp_vals = np.abs(a - b[i]) * scale + c_minus_d
            min_idx = np.argmin(tmp_vals)
            out[i] = min_idx

        # Get the id based on the index.
        selected_id = subset_row.iloc[out].to_numpy()
        tmp_freq_table = Counter(selected_id)
        #counter_weight.update(tmp_freq_table)   
        
        return (w, tmp_freq_table)
    

    
    def _compute_pspop_matching(self):

        """ 
        Compute the pseudo-population using matching approach.
        """

        # We need: 
        # Original GPS value and mean and standard deviation.
        # Original exposure value.
        # Requested exposure level. 
        # Caliper value.
        # Scale value.

        obs_exposure_data = self.data[["id", self.exposure_data_col_name]]
        w_min, w_max = (min(obs_exposure_data[self.exposure_data_col_name]), 
                max(obs_exposure_data[self.exposure_data_col_name]))

        # collect requested exposure level.
        req_exposure = nested_get(self.params, 
                                 ["control_params", 
                                  "bin_seq"])
        
        delta = nested_get(self.params,
                                ["control_params", 
                                "caliper"])

        # check if req_exposure is string
        if isinstance(req_exposure, str):
            req_exposure = eval(req_exposure)

        if req_exposure is None:
            req_exposure = np.arange(w_min+delta/2, w_max, delta)
     
        counter_weight = Counter({key: 0 for key in obs_exposure_data["id"]})
        counter_weight_list = []
 
        n_thread = int(nested_get(self.params, ["run_params", "n_thread"]))

        with ProcessPoolExecutor(max_workers=n_thread) as executor:
            results = list(tqdm(executor.map(self._process_exposure_level, 
                                             req_exposure), 
                                             total=len(req_exposure), 
                                             desc="Processing exposure levels"))
            
        # Process the results and update the counter_weight and counter_weight_list
        for w, tmp_freq_table in results:
            counter_weight_list.append(CounterWeightData(w, tmp_freq_table))
    
        for i_counter_weight in counter_weight_list:
            counter_weight.update(i_counter_weight.counter_weight)
    

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



# def process_exposure_level_2(w, gps_data, data, 
#                                   exposure_data_col_name, params,
#                                   GPS_DENSITY_NORMAL, GPS_DENSITY_KERNEL):

#         # load gps object from gps_data
#         gps_min, gps_max = gps_data.get("gps_minmax")
#         gps_density = gps_data.get("gps_density")
#         if gps_density == GPS_DENSITY_KERNEL:
#             w_resid = gps_data.get("data")["w_resid"]
     
#         # load exposure values
#         obs_exposure_data = data[["id", exposure_data_col_name]]
#         w_min, w_max = (min(obs_exposure_data[exposure_data_col_name]), 
#                         max(obs_exposure_data[exposure_data_col_name]))

#         # controlling parameters
#         delta = nested_get(params,
#                                 ["control_params", 
#                                 "caliper"])
#         scale = nested_get(params,
#                                 ["control_params",
#                                 "scale"])
        
#         dist_measure = nested_get(params, 
#                                  ["control_params", 
#                                   "dist_measure"])

        
#         if gps_density == GPS_DENSITY_NORMAL:
#             p_w = norm.pdf(w, 
#                            gps_data.get("data")["e_gps_pred"], 
#                            gps_data.get("data")["e_gps_std_pred"])
#         elif gps_density == GPS_DENSITY_KERNEL:
#             w_new = ((w - gps_data.get("data")["e_gps_pred"]) 
#                       / gps_data.get("data")["e_gps_std_pred"])
#             p_w = compute_density(w_resid, w)
#         else:
#             raise Exception("GPS model is not defined.")
           

#         # select subset of data that are within the caliper value.
#         subset_idx = np.where(np.abs(
#             obs_exposure_data[exposure_data_col_name] - w) <= delta)[0]
#         subset_row = obs_exposure_data.iloc[subset_idx]["id"]

#         if len(subset_row) == 0:
#             LOGGER.debug(f"No data found within the caliper value ({delta}) " 
#                          f"for the requested exposure level: {w}.")
#             return (w, Counter(None))


#         # standardize GPS and Exposure values.
#         std_w = (w - w_min)/(w_max - w_min)
#         std_gps = (p_w - gps_min)/(gps_max - gps_min)

#         # all data (observational data where w is requested w.)
#         # std_w: scaler
#         # std_gps: vector
#         all_curated_data = pd.DataFrame({"id": obs_exposure_data["id"],
#                                             "std_w": std_w,
#                                             "std_gps": std_gps})
        
#         # subset of data (actual standardized data that are within the caliper value.)
#         # std_w_subset: vector
#         # std_gps_subset: vector
#         std_w_subset = ((obs_exposure_data[obs_exposure_data["id"].
#                         isin(subset_row)][exposure_data_col_name] - w_min)) / (w_max - w_min)
        
#         # TODO: use query.
#         std_gps_subset = (gps_data.get("data")[gps_data.get("data")["id"].isin(subset_row)]["gps_standardized"])

#         # a: subset of data with standardized GPS (std_gps_subset)
#         # b: all data with estimated GPS based on requested exposure level. (std_gps)
#         # c: subset of data with standardized exposure (std_w_subset)
#         # d: the exposure level that is requested. (std_w)
        
#         a = std_gps_subset.to_numpy()
#         b = std_gps
#         c = std_w_subset.to_numpy()
#         d = std_w

#         c_minus_d = abs(c - d)*(1-scale)

#         # compute closest sample from subset of data to these hypothetical samples.
#         # TODO: make this a fucntion
#         len_b = len(b)
#         #len_a = len(a)
#         out = np.zeros(len_b)

#         # chunk_size = int(nested_get(self.params, ["run_params", "chunk_size"]))
        
#         # num_chunks = int(np.ceil(len_b / chunk_size))

#         # with ProcessPoolExecutor(max_workers=n_thread) as executor:
#         #     args_list = [(i * chunk_size, chunk_size, a, b, scale, c_minus_d) for i in range(num_chunks)]
#         #     results = list(executor.map(self.compute_min_idx_proc_chunk, args_list))

#         # # Flatten the results and assign them back to the 'out' array
#         # out = np.concatenate([chunk[chunk != -1] for chunk in results])

    
#         for i in range(len_b):
#             tmp_vals = np.abs(a - b[i]) * scale + c_minus_d
#             min_idx = np.argmin(tmp_vals)
#             out[i] = min_idx

#         # Get the id based on the index.
#         selected_id = subset_row.iloc[out].to_numpy()
#         tmp_freq_table = Counter(selected_id)
#         #counter_weight.update(tmp_freq_table)   
        
#         return (w, tmp_freq_table)
    

# if __name__ == "__main__":

#     from pycausalgps.base.utils import generate_syn_pop
#     from pycausalgps.gps import GeneralizedPropensityScore

#     gps_params = {"gps_density": "normal",
#                   "exposure_column": "treat",
#                   "covariate_column_num": ["cf1", 
#                                            "cf2", 
#                                            "cf3", 
#                                            "cf4", 
#                                            "cf6"],
#                   "covariate_column_cat": ["cf5"],
#                   "libs":{
#                           "xgboost":{
#                                  "n_estimators": 100,
#                                  "max_depth": 3,
#                                  "learning_rate": 0.1,
#                                  "test_rate": 0.2,
#                                  "random_state": 42
#                                  }
#                              }
#     }
    
#     data = generate_syn_pop(sample_size=1000, 
#                             seed_val=456, 
#                             outcome_sd=0.25, 
#                             gps_spec=1, 
#                             cova_spec=2)
        
#     gps = GeneralizedPropensityScore(data, gps_params)
#     results = gps.get_results()
#     gps_data = {
#         'data' : results.get("data"),
#         'gps_density' : results.get("gps_density"),
#         'gps_minmax': results.get("gps_minmax")}

#     # merged_df = pd.merge(data, gps_data, on='id')
#     # merged_df

#     # pspop_params = {"approach" : "weighting", 
#     #                 "exposure_column": "treat",
#     #                 "covariate_column_num": ["cf1", 
#     #                                          "cf2", 
#     #                                          "cf3", 
#     #                                          "cf4", 
#     #                                          "cf6"],
#     #                 "covariate_column_cat": ["cf5"]}
    

#     pspop_params = {"approach" : "matching", 
#                 "exposure_column": "treat",
#                 "covariate_column_num": ["cf1", 
#                                          "cf2", 
#                                          "cf3", 
#                                          "cf4", 
#                                          "cf6"],
#                 "covariate_column_cat": ["cf5"],
#                 "control_params": {"caliper": 1.0,
#                                    "scale": 0.5,
#                                    "dist_measure": "l1",
#                                    "bin_seq": None},
#                 "run_params": {"n_thread": 12,
#                                "chunk_size": 500}}

#     pspop = PseudoPopulation(data=data, 
#                             gps_data=gps_data, 
#                             params=pspop_params)
    

