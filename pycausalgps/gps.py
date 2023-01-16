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
from pycausalgps.database import Database
from pycausalgps.base.utils import nested_get
from pycausalgps.pseudo_population import PseudoPopulation

class GeneralizedPropensityScore:
    """Create a GPS object based on provided parameters.

    GPS object supports continutes treatment with numerical or categorical
    confounders.

    Parameters
    ----------
    project_params : dict
        A dictionary of project parameters. Including:  
            | name: project name  
            | project_id: project id  
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
        self.gps_data = None
        self.gps_minmax = None
        self.hash_value = None
        self.training_report = dict()
        self.pseudo_population_list = list()
        self._connect_to_database()
        self._generate_hash()

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
        try:
            exposure_data = pd.read_csv(exposure_path)
        except Exception as e:
            LOGGER.error(f"Error while loading exposure data: {e}")
            raise e
        LOGGER.debug(f"Exposure data shape: {exposure_data.shape}")

        # load covariate data
        covariate_data = pd.read_csv(covariate_path)
        LOGGER.debug(f"Covariate data shape: {covariate_data.shape}")

        # check data, join them based on the id column. 
        # From the gps params, we know which columns are covariate and which is 
        # exposure.
        if not isinstance(covariate_data, pd.DataFrame):
            raise ValueError("Covariate data is not a pandas DataFrame.")

        if not isinstance(exposure_data, pd.DataFrame):
            raise ValueError("Exposure data is not a pandas DataFrame.")

        if not "id" in covariate_data.columns:
            raise ValueError("Covariate data does not have an 'id' column.")

        if not "id" in exposure_data.columns:
            raise ValueError("Exposure data does not have an 'id' column.")

        # join data based on id column
        data = pd.merge(covariate_data, exposure_data, on="id")

        # check size of data
        if data.shape[0] == 0:
            raise ValueError(f"Joined data size is zero. "
                             + f"(Size of covariate data: {covariate_data.shape[0]}." 
                             + f"Size of exposure data: {exposure_data.shape[0]}.)")

        return data

    def __str__(self):
        return f"GPS object with id: {self.gps_id} \n" +\
               f"GPS parameters: \n {yaml.dump(self.gps_params, default_flow_style=False)} \n" +\
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
            self.gps_params["hash_value"] = self.hash_value
            self.gps_params["gps_id"] = self.gps_id
        else:
            LOGGER.warning("Project hash value is not assigned. " +\
                           "This can happen becuase of running the class individually. " +\
                           "Hash value cannot be generated. ")


    def compute_gps(self):
        """
        Returns Generalized Propensity Score (GPS) value based on input 
        parameters.
        """


        # load data
        _study_data = self.load_data()

        lib_name = nested_get(self.gps_params,
                              ["gps_params", "pred_model", "libs", "name"])

        # lib_name = (self.gps_params.
        #                  get("gps_params").
        #                  get("pred_model").
        #                  get("libs").
        #                  get("name"))

        if lib_name == "xgboost":
            gps_res = self._compute_gps_xgboost(_data=_study_data )

            # check if the keys exist in the gps_res
            for key in ["gps", "e_gps_pred", "e_gps_std", "w_resid"]:
                if not key in gps_res.keys():
                    raise ValueError(f"Key {key} does not exist in the gps_res.")
            
            gps_min, gps_max = gps_res["gps"].min(), gps_res["gps"].max()
            gps_standardized = (gps_res["gps"] - gps_min) / (gps_max - gps_min)    
            self._data = dict(_data=pd.DataFrame(
                                       {"id": _study_data["id"],
                                        "gps": gps_res["gps"],
                                        "gps_standardized": gps_standardized,
                                        "e_gps_pred": gps_res["e_gps_pred"],
                                        "e_gps_std_pred": gps_res["e_gps_std"],
                                        "w_resid": gps_res["w_resid"]}),
                                gps_minmax=[gps_min, gps_max])
           


        else:
            LOGGER.warning(f" GPS computing approach (approach): "+\
                           f" {self.params['approach']}  is not defined.")



    def _compute_gps_xgboost(self, _data: pd.DataFrame) -> dict:

        # select columns that needs to be included.
        # TODO: add checks to make sure that all columns exist.
        # TODO: Move all core functions under base module. The user should be able to compute 
        # different parameters without using the class (provided that all information is given).
        
        num_cols = nested_get(self.gps_params,
                              ["gps_params", "pred_model", 
                               "covariate_column_num"])
        
        num_cols = (self.gps_params.
                         get("gps_params").
                         get("pred_model").
                         get("covariate_column_num"))

        X_num = _data[(self.gps_params.
                            get("gps_params").
                            get("pred_model").
                            get("covariate_column_num"))]

        y = _data[(self.gps_params.
                        get("gps_params").
                        get("pred_model").
                        get("exposure_column"))]
        
        cat_cols = (self.gps_params.
                         get("gps_params").
                         get("pred_model").
                         get("covariate_column_cat"))
        X_cat = _data[cat_cols]

        # Pandas read these comlumns as object.
        for cl in cat_cols:
            X_cat.loc[:,cl] = X_cat.loc[:,cl].astype('category')
        
        if X_num.select_dtypes(include=['object']).shape[1] > 0:
            raise ValueError("Covariate data contains non-numeric columns. ")
    
        # encoding categorical data and merge
        if len(cat_cols) > 0:
            X_cat = pd.get_dummies(_data[cat_cols], columns=cat_cols)
            X_ = X_num
            X_ = X_.join(X_cat)
        else:
            X_ = X_num

        # normalize numerical data
        standard = preprocessing.StandardScaler().fit(X_.loc[:,num_cols])
        # TODO: Added the following line to avoid SettingWithCopyWarning. 
        # Need to check if this is the right way to do it.
        X_ = X_.copy()
        X_.loc[:,num_cols] = standard.transform(X_.loc[:,num_cols])       

        if self.gps_params.get("gps_params").get("model") == "parametric":

            e_gps_pred, training_report = self.xgb_train_it(X_, 
                                                            y.squeeze(), 
                                                            self.gps_params)
            self.training_report.update(training_report)
            e_gps_tmp = (e_gps_pred - y.to_numpy())
            e_gps_std = np.std(e_gps_tmp)
            gps = norm.pdf(y.to_numpy().squeeze(), e_gps_pred, e_gps_std)
            return dict(gps=gps, 
                        e_gps_pred=e_gps_pred, 
                        e_gps_std=e_gps_std,
                        w_resid=None)

        elif self.gps_params.get("gps_params").get("model") == "non-parametric":

            e_gps_pred, training_report = self.xgb_train_it(X_,
                                                            y.squeeze(), 
                                                            self.gps_params)
            target = np.abs(e_gps_pred - y.squeeze())
            e_gps_std_pred, training_report = self.xgb_train_it(X_, 
                                                                target, 
                                                                self.gps_params)
           
            # compute residule
            w_resid = (y.squeeze() - e_gps_pred)/e_gps_std_pred
            
            # compute kernel density estimate ("gaussian")
            kernel = stats.gaussian_kde(w_resid)
            gps = kernel(w_resid)
            return dict(gps=gps,
                        e_gps_pred=e_gps_pred, 
                        e_gps_std=e_gps_std_pred, 
                        w_resid=w_resid)
        else:
            LOGGER.warning(f"gps_model: '{self.params['gps_model']}' is not defined."+\
                           f" Available models: parametric, non-parametric.")
            return dict()

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

    def compute_pseudo_population(self, pspop_params_path):
        """ Computes pseudo population based on the GPS values and 
        pseudo population parameters."""

        # This include loading a yaml file with pseudo population parameters.

        # read pspop_params

        if pspop_params_path is not None:
            try:
                with open(pspop_params_path, 'r') as f:
                    pspop_params = yaml.safe_load(f)
            except Exception as e:
                LOGGER.warning(f"Could not load pspop_params from {pspop_params_path}.")
                LOGGER.warning(e)
                return
        else:
            LOGGER.warning(f"pspop_params_path is not defined.")
            return

        # TODO: compute the combination of the parameters. In this section, if the 
        # user provides a list of approaches, we need to open a new object for each of 
        # them. 

        # read data. 
        # Required data:
        # 1. from gps object: gps, e_gps_pred, e_gps_std, w_resid
        # 2. from pspop_params: pspop_params.get("pspop_params").get("approach")

        # Generate the object to get the hash value.
        ps_pop = PseudoPopulation(self.project_params, self.gps_params, 
                                  pspop_params, self.db_path)

        # check if the ps_pop is already computed. If yes, load it.
        # Why we load from the database? Because we want to avoid recomputing
        # the pseudo population if it is already computed.
        if ps_pop.hash_value in self.pseudo_population_list:
            LOGGER.info(f"Pseudo population is already computed, retireving it from the database.")
            try:
                ps_pop = self.db.get_value(ps_pop.hash_value)
            except Exception as e:
                print(e)
                return
        else:
            # compute ps_pop
            ps_pop.generate_pseudo_population()
            self.pseudo_population_list.append(ps_pop.hash_value)
            self.db.set_value(ps_pop.hash_value, ps_pop)
            


        # if matching, use matching approach. 

        # if weighting, use weighting approach.
        
        # collect the pseudo population.

        # Add the pseudo population to the data.base.

        # Add pseudo population to the pseudo population list.

    def pspop_summary(self):
        """ Prints the summary of the pseudo population."""
        if len(self.pseudo_population_list) == 0:
            print ("The GPS object does not have any pseudo population.")
        else:
            print(f"The GPS object has {len(self.pseudo_population_list)} pseudo population: ")
            for item in self.pseudo_population_list:
                pspop = self.db.get_value(item)
                print(pspop.pspop_id)



    def _connect_to_database(self):
        if self.db_path is None:
            raise Exception("Database is not defined.")
            
        self.db = Database(self.db_path)




    def get_pseudo_population(self, pspop_id):
        """
        Returns the pseudo population object based on the pspop_id.

        Parameters
        ----------
        pspop_id : str
            The id of the pseudo population.

        Returns
        -------
        pspop_obj : PseudoPopulation

        """

        pspop_id_dict = {}
        for pspop_hash in self.pseudo_population_list:
               pspop_obj = self.db.get_value(pspop_hash)
               pspop_id_dict[pspop_obj.pspop_id] = pspop_hash

        if pspop_id in pspop_id_dict.keys():
            return self.db.get_value(pspop_id_dict[pspop_id])
        else:
            print(f"A Pseudo Population object with id:{pspop_id} is not defined.")

if __name__ == "__main__":

    project_params = {"name": "test_project", "id": 136987,
                      "data": {"outcome_path": "data/outcome.csv", 
                               "exposure_path": "data/exposure.csv", 
                               "covariate_path": "data/covariate.csv"}}






    
    
        
        
        
            

    