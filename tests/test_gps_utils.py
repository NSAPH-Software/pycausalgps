import unittest

from pycausalgps.gps_utils import generate_syn_pop

class TestGeneratSynPop(unittest.TestCase):

    def test_data_size(self):

        mydata = generate_syn_pop(150)
        self.assertEqual(len(mydata.data), 150)
        self.assertNotEqual(len(mydata.data), 200)



