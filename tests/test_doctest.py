
import doctest
import unittest

import pycausalgps.database as database
import pycausalgps.gps as gps
import pycausalgps.gps_utils as gps_utils


def test_doctest_suit():
    test_suit = unittest.TestSuite()

    # add tests
    #test_suit.addTest(doctest.DocTestSuite(gps))
    #test_suit.addTest(doctest.DocTestSuite(gps_utils))
    test_suit.addTest(doctest.DocTestSuite(database))
    # set runner
    runner = unittest.TextTestRunner(verbosity=2).run(test_suit)

    assert not runner.failures