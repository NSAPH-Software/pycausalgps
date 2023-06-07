import unittest

import pandas as pd

from pycausalgps.base.utils import nested_get
from pycausalgps.base.utils import generate_syn_pop

class TestUtils(unittest.TestCase):
    def test_nested_get(self):
        d = {"a": {"b": {"c": 1}}}
        keys = ["a", "b", "c"]
        self.assertEqual(nested_get(d, keys), 1)

        d = {"a": {"b": {"c": 1}}}
        keys = ["a", "b", "d"]
        self.assertEqual(nested_get(d, keys), None)
           

class TestGenerateSynPop(unittest.TestCase):

    def test_valid_gps_spec(self):
        sample_size = 100
        seed_val = 42
        outcome_sd = 1
        gps_spec = 3
        cova_spec = 2

        result = generate_syn_pop(sample_size, seed_val, outcome_sd,
                                  gps_spec, cova_spec)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), sample_size)

    def test_invalid_gps_spec(self):
        sample_size = 100
        seed_val = 42
        outcome_sd = 1
        gps_spec = 8  # Invalid gps_spec value
        cova_spec = 2

        with self.assertRaises(ValueError):
            generate_syn_pop(sample_size, seed_val, outcome_sd, 
                             gps_spec, cova_spec)

    def test_valid_cova_spec(self):
        sample_size = 100
        seed_val = 42
        outcome_sd = 1
        gps_spec = 3
        cova_spec = 2

        result = generate_syn_pop(sample_size, seed_val, outcome_sd, 
                                  gps_spec, cova_spec)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), sample_size)

    def test_invalid_cova_spec(self):
        sample_size = 100
        seed_val = 42
        outcome_sd = 1
        gps_spec = 3
        cova_spec = 3  # Invalid cova_spec value

        with self.assertRaises(ValueError):
            generate_syn_pop(sample_size, seed_val, outcome_sd, 
                             gps_spec, cova_spec)

    def test_columns(self):
        sample_size = 100
        seed_val = 42
        outcome_sd = 1
        gps_spec = 3
        cova_spec = 2

        result = generate_syn_pop(sample_size, seed_val, outcome_sd, 
                                  gps_spec, cova_spec)
        expected_columns = ['id', 'Y', 'treat', 'cf1', 'cf2', 'cf3', 
                            'cf4', 'cf5', 'cf6']
        self.assertListEqual(list(result.columns), expected_columns)


if __name__ == '__main__':
    unittest.main()