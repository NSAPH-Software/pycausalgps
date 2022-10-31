"""
gps.py
================================================
The core module for the GPS class.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from .log import LOGGER

class ComputeGPS:
    """Create a GPS object based on provided parameters.

    GPS object supports continutes treatment with numerical or categorical
    confounders.

    Parameters
    ----------
    X : pandas DataFrame
        The dataframe represents confounders, which in the prediction model
        represents input data. Columns should be either 
        numerical or categorical.
    y : pandas DataFrame
        The dataframe represetns treatment, which in the prediction model
        represents target data. 
    params: dictionary
        Includes dictionary of parameters that are required for different
        approaches. The following parameters should be included:
    gps_model: 
        Indicates the type of model for estimating gps value. Available models
        are: parametric, non-parametric.
    approach: 
        Prediction model approach. Implemented model:
            - xgboost

    Notes
    -----
    Other parameters in params input dictionary is related to the selected 
    *gps_model* and *approach*. The following table represents required parameters
    for xgboost.

        **test_size**: 
            The percentage of input data that is used for testing

        **n_estimator**:
            Number of xgboost estimator

        **learning_rate**:
            Learning rate of xgboost. 


    Examples
    --------
    >>> from pycausalgps.gps_utils  import generate_syn_pop
    >>> data = generate_syn_pop(500)
    >>> data['cf5'] = data.cf5.astype('category')
    >>> params = {'approach':"xgboost", 'gps_model': 'parametric', 'test_size': 0.2, 'random_state': 1,'n_estimator': 1000, 'learning_rate': 0.01}
    >>> conf = data[["cf1","cf2","cf3","cf4","cf5","cf6"]]
    >>> treat = data[["treat"]]
    >>> gp = ComputeGPS(X=conf, y=treat, params=params)
    >>> gp.compute_gps()
    >>> len(gp.gps)
    500

    """
    

    def __init__(self, X, y, params) -> None:
        self.X = X
        self.y = y
        self.params = params
        self.gps = None
        self.training_report = None


    def compute_gps(self):
        """
        Returns Generalized Propensity Score (GPS) value based on input 
        parameters.
        """

        if self.params["approach"] == "xgboost":
            self.gps = self._compute_gps_xgboost()
        else:
            LOGGER.warning(f" GPS computing approach (approach): "+\
                           f" {self.params['approach']}  is not defined.")


    def _compute_gps_xgboost(self):

        # Pick columns with categorical data
        cat_cols = list(self.X.select_dtypes(
            include=["category"]).columns.values)

        # Pick columns with numerical data
        num_cols = list(self.X.select_dtypes(
            include=[np.number]).columns.values)
        
        # encoding categorical data and merge
        if len(cat_cols) > 0:
            X_cat = pd.get_dummies(self.X[cat_cols], columns=cat_cols)
            X_ = self.X[num_cols]
            X_ = X_.join(X_cat)
        else:
            X_ = self.X

        # normalize numerical data
        standard = preprocessing.StandardScaler().fit(X_[num_cols])
        X_[num_cols] = standard.transform(X_[num_cols])

        if self.params["gps_model"] == "parametric":

            e_gps_pred = self.xgb_train_it(X_, self.y.squeeze(), self.params)
            e_gps_tmp = (e_gps_pred - self.y.to_numpy())
            e_gps_std = np.std(e_gps_tmp)
            gps = norm.pdf(self.y.to_numpy().squeeze(), e_gps_pred, e_gps_std)
            return (gps)

        elif self.params["gps_model"] == "non-parametric":

            e_gps_pred = self.xgb_train_it(X_, self.y.squeeze(), self.params)
            target = np.abs(e_gps_pred - self.y.squeeze())
            e_gps_std_pred = self.xgb_train_it(X_, target, self.params)
           
            # compute residule
            w_resid = (self.y.squeeze() - e_gps_pred)/e_gps_std_pred
            
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
        xgb = XGBRegressor(n_estimators = params['n_estimator'], 
                            learning_rate = params['learning_rate'])                 
        
        X_train, X_val, y_train, y_val = train_test_split(input, target,
                                    test_size = params['test_size'], 
                                    random_state = params['random_state'])
        # Fit on train data
        xgb.fit(X_train, y_train)

        # Validation
        prediction = xgb.predict(X_val)
        r_s = r2_score(prediction, y_val)
        rmse = np.sqrt(mean_squared_error(prediction, y_val))
        
        training_report = {'r2_score': r_s, 'rmse': rmse}

        #print(f"R2 score: {r_s}, RMSE: {rmse}")

        # Fit on entire data
        predict_all = xgb.predict(input)

        return(predict_all, training_report)


class GPSObject:

    def __init__(self) -> None:
        pass










    
    
        
        
        
            

    