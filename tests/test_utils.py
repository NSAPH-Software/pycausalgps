import unittest

from pycausalgps.base.utils import nested_get

class TestUtils(unittest.TestCase):
    def test_nested_get(self):
        d = {"a": {"b": {"c": 1}}}
        keys = ["a", "b", "c"]
        self.assertEqual(nested_get(d, keys), 1)

        d = {"a": {"b": {"c": 1}}}
        keys = ["a", "b", "d"]
        self.assertEqual(nested_get(d, keys), None)
           