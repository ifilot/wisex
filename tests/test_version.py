import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import wisex

class TestVersion(unittest.TestCase):

    def test_version(self):
        strver = wisex.__version__.split('.')
        self.assertTrue(strver[0].isnumeric())
        self.assertTrue(strver[1].isnumeric())
        self.assertTrue(strver[2].isnumeric())

if __name__ == '__main__':
    unittest.main()