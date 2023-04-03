
import doctest
import unittest

import pycausalgps.gps as gps



def test_doctest_suit():
    test_suit = unittest.TestSuite()

    # add tests
    test_suit.addTest(doctest.DocTestSuite(gps))
    
    # set runner
    runner = unittest.TextTestRunner(verbosity=2).run(test_suit)

    assert not runner.failures