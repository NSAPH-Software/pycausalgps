"""
gps.py
================================================
The core module for the GPS class.
"""

from random import random
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import norm
from .log import LOGGER

class GPS:
    

    def __init__(self, X, y, params) -> None:
        self.X = X
        self.y = y
        self.params = params


    def compute_gps(self):

        if self.params["approach"] == "xgboost":
            if self.params["gps_model"] == "parametric":
                return self._compute_gps_xgboost_parametric()
            elif self.params['gps_model'] == 'non-parametric':
                return self._compute_gps_xgboost_nonparametric()
            else:
                LOGGER.warning(f"gps_model '{self.params['gps_model']}' is not valid."+\
                               f" Valid options: parametric, non-parametric")
        else:
            LOGGER.warning(f" GPS computing approach (approach): {self.params['approach']}  is not defined.")
            # compute GPS value

            # Data is either numerical or category
            # TODO: Double-check other possiblities
    
            # Pick columns with numerical and categorical data
            cat_cols = []
            for col in self.X:
                if self.X[col].dtypes == "category":
                    cat_cols.append(col)
            cols = list(self.X.columns)
            num_cols = [x for x in cols if x not in cat_cols]
    
            # encoding categorical data and merge
            if len(cat_cols) > 0:
                X_cat = pd.get_dummies(self.X[cat_cols], columns=cat_cols)
                X_ = self.X[num_cols]
                X_ = X_.join(X_cat)
            else:
                X_ = self.X
    
            # normalize data
            standard = preprocessing.StandardScaler().fit(X_[num_cols])
            X_[num_cols] = standard.transform(X_[num_cols])

            # Split data into training and testing
            X_train, X_val, y_train, y_val = train_test_split(X_, self.y,
                                             test_size = self.params(test_size), 
                                             random_state = self.params(random_state))
            
            #  Create prediction mode
            xgb = XGBRegressor(n_estimators = self.params(n_estimator), 
                               learning_rate = self.params(learning_rate))
            
            # Fit on train data
            xgb.fit(X_train, y_train)

            # estimate performance on validation data
            prediction = xgb.predict(X_val)
            r_s = r2_score(prediction, y_val)
            print(f"R2 score: {r_s}")

            rmse = np.sqrt(mean_squared_error(prediction, y_val))

            # estimate gps value
            parametric = True

            
            e_gps = xgb.predict(X_)
            e_gps_tmp = (e_gps - self.y)
            e_gps_std = e_gps_tmp.std()
            gps = norm.pdf(self.y, e_gps, e_gps_std)
            
            return(gps)

    def _compute_gps_xgboost_parametric(self):

        if self.params["approach"] == "xgboost":
            # compute GPS value

            # Data is either numerical or category
            # TODO: Double-check other possiblities
    
            # Pick columns with numerical and categorical data
            cat_cols = []
            for col in self.X:
                if self.X[col].dtypes == "category":
                    cat_cols.append(col)
            cols = list(self.X.columns)
            num_cols = [x for x in cols if x not in cat_cols]
    
            # encoding categorical data and merge
            if len(cat_cols) > 0:
                X_cat = pd.get_dummies(self.X[cat_cols], columns=cat_cols)
                X_ = self.X[num_cols]
                X_ = X_.join(X_cat)
            else:
                X_ = self.X
    
            # normalize data
            standard = preprocessing.StandardScaler().fit(X_[num_cols])
            X_[num_cols] = standard.transform(X_[num_cols])

            # Split data into training and testing
            X_train, X_val, y_train, y_val = train_test_split(X_, self.y,
                                             test_size = test_size, 
                                             random_state=random_state)
            
            #  Create prediction mode
            xgb = XGBRegressor(n_estimators = n_estimator, 
                               learning_rate = learning_rate)
            
            # Fit on train data
            xgb.fit(X_train, y_train)

            # estimate performance on validation data
            prediction = xgb.predict(X_val)
            r_s = r2_score(prediction, y_val)
            rmse = np.sqrt(mean_squared_error(prediction, y_val))
            print(f"R2 score: {r_s}, RMSE: {rmse}")


            # estimate gps value
            e_gps = xgb.predict(X_)
            e_gps_tmp = (e_gps - self.y.to_numpy())
            e_gps_std = np.std(e_gps_tmp)
            gps = norm.pdf(self.y.to_numpy().squeeze(), e_gps, e_gps_std)
            
            return(gps)

    def _compute_gps_xgboost_nonparametric(self):
        print("Is not implemented.")












    
    
        
        
        
            

    