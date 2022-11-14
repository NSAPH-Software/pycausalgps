"""
pseudo_population.py
====================
The core module for the PseudoPopulation class.
"""

import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec


from pycausalgps.log import LOGGER
from pycausalgps.database import Database
from pycausalgps.rscripts.rfunctions import (compute_density,
                                             compute_absolute_weighted_corr)

class PseudoPopulation:

    def __init__(self, project_params, gps_params, 
                       pspop_params, db_path) -> None:
        self.project_params = project_params
        self.gps_params = gps_params
        self.pspop_params = pspop_params
        self.db_path = db_path
        self.hash_value = None
        self.pspop_id = None
        self.counter_weight = None
        self.covariate_balance = None
        self._generate_hash()
        self._connect_to_database()


    def _generate_hash(self) -> None:
        """
        Generate a hash value for the pseudo-population object.
        """

        hash_string = json.dumps(self.pspop_params, sort_keys=True)

        if self.gps_params.get("hash_value"):
            hash_string = self.gps_params.get("hash_value") + hash_string
            self.hash_value = hashlib.sha256(
                hash_string.encode("utf-8")).hexdigest()
            # generating random id for the pseudo-population
            self.pspop_id = hashlib.shake_256(self.hash_value.encode()).hexdigest(8)
            self.pspop_params["hash_value"] = self.hash_value
        else:
            LOGGER.warning("Project hash value is not assigned. " +\
                           "This can happen becuase of running the class individually. " +\
                           "Hash value cannot be generated. ")

            
    def _connect_to_database(self):
        """
        Connect to the database.
        """
        if self.db_path is None:
            raise Exception("Database is not defined.")
            
        self.db = Database(self.db_path)
    
    def generate_pseudo_population(self) -> None:
        """
        Compute the pseudo-population.
        """
        
        # For computing pspop we need to have the following:
        # 1. gps_params (gps values, estimated exposure, estimated exposure std,
        #                w_resid)
        # 2. pspop_params (approach, caliper, scale, 

        if self.pspop_params.get("pspop_params").get("approach") == "weighting":
            counter_weight = self._compute_pspop_weighting()
            self.counter_weight = counter_weight
            self._compute_covariate_balance()
        elif self.pspop_params.get("pspop_params").get("approach") == "matching":
            self._compute_pspop_matching()
        else:
            raise Exception("Approach is not defined.")


    def load_exposure(self) -> None:
        """
        Load the exposure values.
        """
        exposure_path = self.project_params.get("data").get("exposure_path")

        # load exposure values.
        exposure_data = pd.read_csv(exposure_path)
        LOGGER.debug(f"Exposure data shape: {exposure_data.shape}")
        
        return exposure_data

    def load_confounders(self) -> None:
        """
        Load the confounder values.
        """
        covariate_path = self.project_params.get("data").get("covariate_path")

        # load confounder values.
        covariate_data = pd.read_csv(covariate_path)
        LOGGER.debug(f"Confounder data shape: {covariate_data.shape}")
        
        # select the comlumns that are used in the gps
        covariate_col_num = self.gps_params.get("gps_params").get("pred_model").get("covariate_column_num")
        covariate_col_cat = self.gps_params.get("gps_params").get("pred_model").get("covariate_column_cat")
        
        if covariate_col_num is not None:
            covariate_data_num = covariate_data[covariate_col_num]
        else:
            covariate_data_num = None
        
        if covariate_col_cat is not None:
            covariate_data_cat = covariate_data[covariate_col_cat]
        else:
            covariate_data_cat = None


        return covariate_data_num, covariate_data_cat


    def _compute_pspop_weighting(self) -> None:
        """
        Compute the pseudo-population using weighting.
        """
        
        # The weight of each sample is equal to the probablity of getting that exposure
        # in the data over the probablity of getting that exposure given the covariates (GPS).

        # temp test
        compute_density_with_r = True
        
        # load exposure values. 
        exposure_data = self.load_exposure()

        exposure_data_col_name = self.gps_params.get("gps_params").get("pred_model").get("exposure_column")

        # compute density of the exposure in the data.
        if compute_density_with_r:
            # compute density with R.
            w_val = exposure_data[exposure_data_col_name].to_numpy()
            data_density = compute_density(w_val, w_val)
            print("Density was computed with R.")
        else:
            exp_data = exposure_data[exposure_data_col_name]
            kde = gaussian_kde(exp_data)
            data_density = kde(exp_data)
        
 
        # Extract the gps object from the database.
        gps_obj = self.db.get_value(self.gps_params.get("hash_value"))

        ipw = data_density / gps_obj.gps
        
        return ipw
        
        


    def _compute_pspop_matching(self) -> None:
        pass



    def _compute_covariate_balance(self) -> None:
        """
        Compute covariate balance.
        """
        if self.counter_weight is None:
            raise Exception("Counter weight is not defined. " +\
                "Try generating the pseudo-population first.")

        # load exposure values.
        exposure_data = self.load_exposure()
        exp_data = exposure_data[self.gps_params.get("gps_params").get("pred_model").get("exposure_column")].to_numpy()

        # load the confounders.
        covariate_data_num, covariate_data_cat = self.load_confounders()

        # compute weighted correlation between the confounders and the exposure.
        covariate_balance = compute_absolute_weighted_corr(exp_data, 
                                                           self.counter_weight, 
                                                           covariate_data_num,
                                                           covariate_data_cat) 

        original_data_covariate_balance = compute_absolute_weighted_corr(exp_data,
                                                                         np.ones(self.counter_weight.shape),
                                                                         covariate_data_num,
                                                                         covariate_data_cat)                    

        cov_balance_current = covariate_balance
        cov_balance_current.rename(columns = {'value':'current'}, inplace = True)
        cov_balance_original = original_data_covariate_balance
        cov_balance_original.rename(columns = {'value':'original'}, inplace = True)


        cov_balance = pd.merge(cov_balance_current, cov_balance_original, on='name')

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
        ax3.text(xlim3[0], ylim3[1]-0.15, 
                 f'GPS Object ID: {self.gps_params.get("gps_id")}', 
                 **hyper_param_3)
        ax3.text(xlim3[0], ylim3[1]-0.2,
                    f'GPS hash value: {self.gps_params.get("hash_value")}',
                    **hyper_param_3)
        ax3.text(xlim3[0], ylim3[1]-0.25,
                    f'Pseudo population ID: {self.pspop_id}',
                    **hyper_param_3)
        ax3.text(xlim3[0], ylim3[1]-0.3,
                    f'Pseudo population hash value: {self.hash_value}',
                    **hyper_param_3)

        fig.suptitle("Covariate Balance", fontsize=16)
        plt.show()   