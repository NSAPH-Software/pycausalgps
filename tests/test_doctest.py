
import doctest
import unittest

import pycausalgps.dataset as dataset
import pycausalgps.gps_utils as gps_utils


def test_doctest_suit():
    test_suit = unittest.TestSuite()

    # add tests
    test_suit.addTest(doctest.DocTestSuite(dataset))
    test_suit.addTest(doctest.DocTestSuite(gps_utils))
    
    # set runner
    runner = unittest.TextTestRunner(verbosity=2).run(test_suit)

    assert not runner.failures