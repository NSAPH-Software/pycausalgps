"""
gps.py
======
The core module for the GeneralizedPropensityScore class.
"""

import json
import yaml
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from pycausalgps.log import LOGGER

class GeneralizedPropensityScore:
    """Create a GPS object based on provided parameters.

    GPS object supports continutes treatment with numerical or categorical
    confounders.

    Parameters
    ----------
    project_params : dict
        A dictionary of project parameters. Including:  
        | name: project name  
        | id: project id  
        | data: a dictionary of data parameters. Including:  
            | exposure_path: path to exposure data
            | covariate_path: path to covariate data
            | outcome_path: path to outcome data
    
    gps_params : dict
        A dictionary of gps parameters.

    db_path : str
        Path to the database.
    """
    

    def __init__(self, project_params, gps_params, db_path) -> None:
        self.project_params = project_params
        self.gps_params = gps_params
        self.db_path = db_path
        self.gps_id = None
        self.gps = None
        self.hash_value = None
        self._generate_hash()
        self.training_report = dict()
        self.pseudo_population_list = list()

    def load_data(self):
        """
        Load data for GPS computation.
        TODO: add support for streaming large data.
        TODO: We need to load data into class and have sufficient methods to
        work with them. 
        """
        
        # path to exposure data
        exposure_path = self.project_params.get("data").get("exposure_path")
        
        # path to covariate data
        covariate_path = self.project_params.get("data").get("covariate_path")

        # load exposure data
        exposure_data = pd.read_csv(exposure_path)
        LOGGER.debug(f"Exposure data shape: {exposure_data.shape}")

        # load covariate data
        covariate_data = pd.read_csv(covariate_path)
        LOGGER.debug(f"Covariate data shape: {covariate_data.shape}")

        #TODO: check data.

        return exposure_data, covariate_data


    def __str__(self):
        return f"GPS object with id: {self.gps_id} \n" +\
               f"GPS parameters: {yaml.dump(self.gps_params, default_flow_style=False)} \n" +\
               f" ----- *** -----  \n" +\
               f"Training reports: \n {yaml.dump(self.training_report, default_flow_style=False)} \n"



    def __repr__(self):
        return f"GeneneralizedPropensityScore({self.project_params}),{self.gps_params})"

    def _generate_hash(self):
        """
        Generate hash for the gps object.
        """
        hash_string = json.dumps(self.gps_params, sort_keys=True)

        if self.project_params.get("hash_value"):
            hash_string = self.project_params.get("hash_value") + hash_string
            self.hash_value = hashlib.sha256(
                hash_string.encode('utf-8')).hexdigest()
            # generating random id for gps object by a short hash
            self.gps_id = hashlib.shake_256(self.hash_value.encode()).hexdigest(8)      
        else:
            LOGGER.warning("Project hash value is not assigned. " +\
                           "This can happen becuase of running the class individually")


    def compute_gps(self):
        """
        Returns Generalized Propensity Score (GPS) value based on input 
        parameters.
        """

        # load data
        exposure_data, covariate_data = self.load_data()

        lib_name = self.gps_params.get("gps_params").get("pred_model").get("libs").get("name")

        if lib_name == "xgboost":
            self.gps = self._compute_gps_xgboost(X = covariate_data, 
                                                 y = exposure_data)
        else:
            LOGGER.warning(f" GPS computing approach (approach): "+\
                           f" {self.params['approach']}  is not defined.")



    def _compute_gps_xgboost(self, X, y):

        # select columns that needs to be included.
        # TODO: add checks to make sure that all columns exist.
        X = X[self.gps_params.get("gps_params").get("pred_model").get("covariate_column")]
        y = y[self.gps_params.get("gps_params").get("pred_model").get("exposure_column")]
        
        
        # Pick columns with categorical data
        cat_cols = list(X.select_dtypes(
            include=["category"]).columns.values)

        # Pick columns with numerical data
        num_cols = list(X.select_dtypes(
            include=[np.number]).columns.values)
        
        # encoding categorical data and merge
        if len(cat_cols) > 0:
            X_cat = pd.get_dummies(X[cat_cols], columns=cat_cols)
            X_ = X[num_cols]
            X_ = X.join(X_cat)
        else:
            X_ = X

        # normalize numerical data
        standard = preprocessing.StandardScaler().fit(X_[num_cols])
        X_[num_cols] = standard.transform(X_[num_cols])

        if self.gps_params.get("gps_params").get("model") == "parametric":

            e_gps_pred, training_report = self.xgb_train_it(X_, y.squeeze(), self.gps_params)
            self.training_report.update(training_report)
            e_gps_tmp = (e_gps_pred - y.to_numpy())
            e_gps_std = np.std(e_gps_tmp)
            gps = norm.pdf(y.to_numpy().squeeze(), e_gps_pred, e_gps_std)
            return (gps)

        elif self.gps_params.get("gps_params").get("model") == "non-parametric":

            e_gps_pred = self.xgb_train_it(X_, y.squeeze(), self.gps_params)
            target = np.abs(e_gps_pred - y.squeeze())
            e_gps_std_pred = self.xgb_train_it(X_, target, self.gps_params)
           
            # compute residule
            w_resid = (y.squeeze() - e_gps_pred)/e_gps_std_pred
            
            # compute kernel density estimate ("gaussian")
            kernel = stats.gaussian_kde(w_resid)
            gps = kernel(w_resid)
            return(gps)
        else:
            LOGGER.warning(f"gps_model: '{self.params['gps_model']}' is not defined."+\
                           f" Available models: parametric, non-parametric.")

    @staticmethod
    def xgb_train_it(input, target, params):
        """ Creates XGBoost regressor model and returns predicted value for
        all input data."""

        #initiate the model
        # TODO: collect library hyperparameters via dictionary and the main key 
        # can be the library name + hyperparams.
        # TODO: check hyper params before feeding to the model. If it is not defined, 
        # the model will use default values which is misleading.
        xgb = XGBRegressor(n_estimators = params.get("gps_params").get("pred_model").get("libs").get("n_estimators"),
                           learning_rate = params.get("gps_params").get("pred_model").get("libs").get("learning_rate"))                 
        
        X_train, X_val, y_train, y_val = train_test_split(input, target,
                                    test_size = params.get("gps_params").get("pred_model").get("libs").get("test_rate"), 
                                    random_state = params.get("gps_params").get("pred_model").get("libs").get("random_state"))
        # Fit on train data
        xgb.fit(X_train, y_train)

        # Validation
        prediction = xgb.predict(X_val)
        r_s = r2_score(prediction, y_val)
        rmse = np.sqrt(mean_squared_error(prediction, y_val))
        
        training_report = {'r2_score': float(r_s),
                           'rmse': float(rmse)}

        #print(f"R2 score: {r_s}, RMSE: {rmse}")

        # Fit on entire data
        predict_all = xgb.predict(input)

        return  predict_all, training_report

    def compute_pseudo_population(self, ps_pop_params):
        """ Computes pseudo population based on the GPS values and 
        pseudo population parameters."""

        # read ps_pop_params

        # read data. 

        # check data. 

        # if matching, use matching approach. 

        # if weighting, use weighting approach.
        
        # collect the pseudo population.

        # Add the pseudo population to the data.base.

        # Add pseudo population to the pseudo population list.



if __name__ == "__main__":

    project_params = {"name": "test_project", "id": 136987,
                      "data": {"outcome_path": "data/outcome.csv", 
                               "exposure_path": "data/exposure.csv", 
                               "covariate_path": "data/covariate.csv"}}






    
    
        
        
        
            

    