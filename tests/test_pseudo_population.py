import unittest
from collections import Counter

import numpy as np
import pandas as pd

from pycausalgps.pseudo_population import PseudoPopulation, CounterWeightData


class TestPseudoPopulation(unittest.TestCase):

    def setUp(self):

        self.data = pd.DataFrame({
                        "id": range(1, 101),
                        "exposure": np.random.random(100),
                        "num_feature_1": np.random.random(100),
                        "num_feature_2": np.random.random(100),
                        "cat_feature": np.random.choice(["A", "B", "C"], 100)})
        
        gps_data = pd.DataFrame({
                        "id": range(1, 101),
                        "gps": np.random.random(100),
                        "gps_standardized": np.random.random(100),
                        "e_gps_pred": np.random.random(100),
                        "e_gps_std_pred": np.random.random(100),
                        "w_resid": np.random.random(100)})
        
        self.gps_data = {
            'data' : gps_data,
            'gps_density' : "normal",
            'gps_minmax': [min(gps_data['gps']), max(gps_data['gps'])]
        }

        self.params_weighting = {
            'approach': PseudoPopulation.APPROACH_WEIGHTING,
            'exposure_column': 'exposure',
            'covariate_column_num': ['num_feature_1', 'num_feature_2'],
            'covariate_column_cat': ['cat_feature']}
        
        self.params_matching = {
            'approach': PseudoPopulation.APPROACH_MATCHING,
            'exposure_column': 'exposure',
            'covariate_column_num': ['num_feature_1', 'num_feature_2'],
            'covariate_column_cat': ['cat_feature'], 
            "control_params": {"caliper": 1.0,
                                   "scale": 0.5,
                                   "dist_measure": "l1",
                                   "bin_seq": None},
            "run_params": {"n_thread": 12,
                               "chunk_size": 500}}
        

    def test_init_weighting(self):
        pspop = PseudoPopulation(self.data, 
                                 self.gps_data, 
                                 self.params_weighting)
        results = pspop.get_results()

        self.assertIn("data", results)
        self.assertIn("params", results)
        self.assertIn("compiling_report", results)
        self.assertIn("covariate_balance", results)
        self.assertEqual(results.get("data").shape[0], 100)
        self.assertEqual(results.get("covariate_balance").shape[0], 3)

    def test_init_matching(self):
        pspop = PseudoPopulation(self.data, self.gps_data, self.params_matching)
        results = pspop.get_results()

        self.assertIn("data", results)
        self.assertIn("params", results)
        self.assertIn("compiling_report", results)
        self.assertIn("covariate_balance", results)
        self.assertEqual(results.get("data").shape[0], 100)
        self.assertEqual(results.get("covariate_balance").shape[0], 3)


    def test_raise_exception_when_counter_weight_not_defined(self):
        pspop = PseudoPopulation(self.data, self.gps_data, self.params_matching)
        pspop.counter_weight = None
        with self.assertRaises(Exception):
            pspop._compute_covariate_balance()

    def test_merge_data_creattion(self):
        self.pspop_weighting = PseudoPopulation(self.data, 
                                                self.gps_data, 
                                                self.params_weighting)
        self.pspop_matching = PseudoPopulation(self.data, 
                                               self.gps_data, 
                                               self.params_matching)
        self.pspop_weighting._compute_covariate_balance()
        self.assertIsNotNone(self.pspop_weighting.merged_data)

        self.pspop_matching._compute_covariate_balance()
        self.assertIsNotNone(self.pspop_matching.merged_data)


    def test_compute_covariate_balance_values(self):

        self.pspop_weighting = PseudoPopulation(self.data, 
                                                self.gps_data, 
                                                self.params_weighting)
        self.pspop_matching = PseudoPopulation(self.data, 
                                               self.gps_data, 
                                               self.params_matching)
        cov_balance_current = self.pspop_weighting.covariate_balance['current']
        cov_balance_original = self.pspop_weighting.covariate_balance['original']
        self.assertTrue(np.all(np.abs(cov_balance_current) <= 1))
        self.assertTrue(np.all(np.abs(cov_balance_original) <= 1))

        cov_balance_current = self.pspop_matching.covariate_balance['current']
        cov_balance_original = self.pspop_matching.covariate_balance['original']
        self.assertTrue(np.all(np.abs(cov_balance_current) <= 1))
        self.assertTrue(np.all(np.abs(cov_balance_original) <= 1))

    def test_covariate_balance_data_frame(self):
        self.pspop_weighting = PseudoPopulation(self.data, 
                                                self.gps_data, 
                                                self.params_weighting)
        self.pspop_matching = PseudoPopulation(self.data, 
                                               self.gps_data, 
                                               self.params_matching)
        expected_columns = ['name', 'current', 'original']
        self.assertTrue(
            set(expected_columns).issubset(
            self.pspop_weighting.covariate_balance.columns))
        self.assertTrue(
            set(expected_columns).issubset(
            self.pspop_matching.covariate_balance.columns))







if __name__ == "__main__":
    unittest.main()