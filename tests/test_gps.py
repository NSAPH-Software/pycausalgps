import unittest
import numpy as np
import pandas as pd
from pycausalgps.gps import GeneralizedPropensityScore

class TestGeneralizedPropensityScore(unittest.TestCase):

    def setUp(self):
        # Prepare test data and parameters for the GeneralizedPropensityScore class
        self.data = pd.DataFrame({"id": range(1, 101),
                                  "exposure": np.random.random(100),
                                  "num_feature_1": np.random.random(100),
                                  "num_feature_2": np.random.random(100),
                                  "cat_feature": np.random.choice(["A", "B", "C"], 100)})

        self.params = {
            "gps_params": {
                "model": "parametric",
                "exposure_column": ["exposure"],
                "covariate_column_num": ["num_feature_1", "num_feature_2"],
                "covariate_column_cat": ["cat_feature"],
                "libs": {
                    "xgboost": {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 3,
                        "test_rate": 0.2,
                        "random_state": 42
                    }
                }
            }
        }

    def test_init(self):
        gps = GeneralizedPropensityScore(self.data, self.params)
        self.assertEqual(len(gps.data), 100)
        self.assertIsInstance(gps.params, dict)

    def test_data_setter(self):
        gps = GeneralizedPropensityScore(self.data, self.params)
        with self.assertRaises(ValueError):
            gps.data = "invalid_data_type"

    def test_params_setter(self):
        gps = GeneralizedPropensityScore(self.data, self.params)
        with self.assertRaises(ValueError):
            gps.params = "invalid_params_type"

    def test_compute_gps_xgboost(self):
        gps = GeneralizedPropensityScore(self.data, self.params)
        results = gps.get_results()
        self.assertIn("gps", results["data"].columns)
        self.assertIn("gps_standardized", results["data"].columns)
        self.assertIn("e_gps_pred", results["data"].columns)
        self.assertIn("e_gps_std_pred", results["data"].columns)
        self.assertIn("w_resid", results["data"].columns)


if __name__ == "__main__":
    unittest.main()