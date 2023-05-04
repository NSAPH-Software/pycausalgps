"""
gps.py
======
The core module for the GeneralizedPropensityScore class.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from pycausalgps.log import LOGGER
from pycausalgps.base.utils import nested_get

class GeneralizedPropensityScore:
    """
    Create a GPS object based on provided parameters.

    GPS object supports continuous treatment with numerical or categorical
    confounders.

    Parameters
    ----------

    data: pd.DataFrame
        A pandas DataFrame that contains the data for the GPS computation.
    params : dict
        A dictionary of gps parameters. This includes the folowing required
        parameters:

            - gps_density: str
                A string that indicates the density estimation method to be
                used. Available options are "normal" and "kernel".
            - exposure_column: str  
                A string that indicates the name of the exposure column.
            - covariate_column_num: list
                A list of strings that indicates the names of the numerical
                covariate columns.
            - covariate_column_cat: list
                A list of strings that indicates the names of the categorical
                covariate columns.
            - libs: dict
                A dictionary of libraries and their parameters. Currently only
                xgboost is supported. Any parameters for hypertuning the 
                xgboost model can be passed including:

                    - n_estimators: int
                        Number of trees to fit.
                    - learning_rate: float
                        Learning rate for the xgboost model.
                    - max_depth: int
                        Maximum depth of a tree.
                    - test_rate: float
                        The proportion of the data to be used for testing.
                    - random_state: int
                        Random state for the xgboost model.

    Returns
    -------

    GeneralizedPropensityScore object that includes the following attributes:
        - data: pd.DataFrame
            The gps and auxilary columns.
        - gps_minmax: list
            A list of the minimum and maximum gps values.
        - training_report: dict
            A dictionary of the training report for the xgboost model.
    """
    
    def __init__(self, data: pd.DataFrame, params: dict) -> None:
        self.data = data
        self.params = params
        self.training_report = {}
        self.results = None
        self._check_data()
        self._check_params()
        self._compute_gps()

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
        if not isinstance(value, dict):
            raise ValueError("Params must be a dictionary.")
        self.__params = value


    def __str__(self) -> str:
        return f"GeneralizedPropensityScore({len(self.data)} rows)"

    def __repr__(self) -> str:
        return (f"GeneralizedPropensityScore(data={self.data},"
                f"params={self.params})")
    

    def _check_data(self):
        """
        Check if the data is in the correct format.
        """
        #TODO: add more checks
        pass

    def _check_params(self):
        """
        Check if the params are in the correct format.
        """

        required_params = ["gps_density", "exposure_column", 
                           "covariate_column_num", "covariate_column_cat", 
                           "libs"]
        
        for param in required_params:
            if not param in self.params.keys():
                raise ValueError(f"Required parameter {param} is missing.")

    def compute_gps(self) -> None:
        """
        Returns Generalized Propensity Score (GPS) value based on input 
        parameters.
        """

        libs = nested_get(self.params, ["libs"])
        
        first_level_keys = list(libs.keys())
        
        if len(first_level_keys) > 1:
            raise ValueError("Multiple libraries are not supported yet.")
        
        lib_name = first_level_keys[0]

        if lib_name == "xgboost":
            gps_res = self.compute_gps_xgboost()

            # check if the keys exist in the gps_res
            for key in ["gps", "e_gps_pred", "e_gps_std", "w_resid"]:
                if not key in gps_res.keys():
                    raise ValueError(f"Key {key} does not exist in "
                                     f"the gps_res.")
            
            gps_min, gps_max = gps_res["gps"].min(), gps_res["gps"].max()
            gps_standardized = (gps_res["gps"] - gps_min) / (gps_max - gps_min)    
            results = {"data":pd.DataFrame(
                                       {"id": self.data["id"],
                                        "gps": gps_res["gps"],
                                        "gps_standardized": gps_standardized,
                                        "e_gps_pred": gps_res["e_gps_pred"],
                                        "e_gps_std_pred": gps_res["e_gps_std"],
                                        "w_resid": gps_res["w_resid"]}),
                        "gps_minmax":[gps_min, gps_max],
                        "gps_density": self.params["gps_density"],
                        "training_report": self.training_report,
            }
            return results

        else:
            LOGGER.warning(f" GPS computing approach (approach): "
                           f" {self.params['approach']}  is not defined.")
            return None


    def _compute_gps(self) -> None:
        """
        Compute GPS based on the provided parameters.
        """
        self.results = self.compute_gps()

    def compute_gps_xgboost(self) -> dict:
        """
        Compute GPS using XGBoost library.
        """

        # select columns that needs to be included.
        # TODO: add checks to make sure that all columns exist.
        # TODO: Move all core functions under base module. The user should be able to compute 
        # different parameters without using the class (provided that all information is given).
        
        # preprocess data   
        num_cols = nested_get(self.params, ["covariate_column_num"])
        
        X_num = self.data[num_cols]

        exposure_col = nested_get(self.params, ["exposure_column"])

        y = self.data[exposure_col]
 
        cat_cols = nested_get(self.params, ["covariate_column_cat"])

        X_cat = self.data[cat_cols]
        X_cat = X_cat.copy()

        # Pandas read these comlumns as object.
        for cl in cat_cols:
            X_cat.loc[:,cl] = X_cat.loc[:,cl].astype('category')
        
        if X_num.select_dtypes(include=['object']).shape[1] > 0:
            raise ValueError("Covariate data contains non-numeric columns. ")
    
        # encoding categorical data and merge
        if len(cat_cols) > 0:
            X_cat = pd.get_dummies(self.data[cat_cols], columns=cat_cols)
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

        # get hyperparameters
        hyper_params = nested_get(self.params, ["libs", "xgboost"])


        if self.params.get("gps_density") == "normal":

            e_gps_pred, training_report = self.xgb_train_it(X_, 
                                                            y.squeeze(), 
                                                            hyper_params)
            self.training_report["training_model"] = training_report
            e_gps_tmp = (e_gps_pred - y.to_numpy())
            e_gps_std = np.std(e_gps_tmp)
            gps = norm.pdf(y.to_numpy().squeeze(), e_gps_pred, e_gps_std)
            return dict(gps=gps, 
                        e_gps_pred=e_gps_pred, 
                        e_gps_std=e_gps_std,
                        w_resid=None)

        elif self.params.get("gps_density") == "kernel":

            e_gps_pred, training_report = self.xgb_train_it(X_,
                                                            y.squeeze(), 
                                                            hyper_params)
            
            self.training_report["training_model"] = training_report
            target = np.abs(e_gps_pred - y.squeeze())
            e_gps_std_pred, training_report = self.xgb_train_it(X_, 
                                                                target, 
                                                                hyper_params)
            
            self.training_report["training_noise"] = training_report
            
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
            LOGGER.warning(f"gps_density: '{self.params['gps_density']}'"
                           f" is not defined."
                           f" Available options: normal, kernel.")
            return dict()

    @staticmethod
    def xgb_train_it(X: pd.DataFrame, 
                     target: pd.Series, 
                     params: dict) -> tuple:
        """ Create XGBoost regressor model 
        
        Parameters
        ----------
        X : pd.DataFrame
            Covariate data
        target : pd.Series
            Exposure data
        params : dict
            Dictionary of parameters
        
        Returns
        -------
        predict_all : np.ndarray
            Predicted exposure for all data
        training_report : dict
            Dictionary of training report
        """
        
        LOGGER.debug(f"XGBoost provided parameters: {params}")

        # Collect hyperparameters
        n_estimators = params.get("n_estimators", 100)
        learning_rate = params.get("learning_rate", 0.1)
        test_size = params.get("test_rate", 0.2)
        max_depth = params.get("max_depth", 3)
        random_state = params.get("random_state", 42)
        n_jobs = params.get("n_jobs", 1)

        LOGGER.debug(f"XGBoost used parameters: n_estimators={n_estimators}, "
                     f"learning_rate={learning_rate}, test_size={test_size}, "
                     f"random_state={random_state}, "
                     f"max_depth={max_depth}.")

        xgb = XGBRegressor(n_estimators = n_estimators,
                           learning_rate = learning_rate,
                           max_depth = max_depth,
                           n_jobs=n_jobs)                 
        
        X_train, X_val, y_train, y_val = train_test_split(X, target,
                                    test_size = test_size, 
                                    random_state = random_state)
        # Fit on train data
        xgb.fit(X_train, y_train)

        # Validation
        prediction = xgb.predict(X_val)
        r_s = r2_score(prediction, y_val)
        rmse = np.sqrt(mean_squared_error(prediction, y_val))
        
        training_report = {'r2_score': float(r_s),
                           'rmse': float(rmse)}

        # Fit on entire data
        predict_all = xgb.predict(X)

        return  predict_all, training_report
    
    def get_results(self) -> dict:
        """ Return results of GPS computation
        
        Returns
        -------
        dict
            Dictionary of results
        """
        return self.results






    
    
        
        
        
            

    