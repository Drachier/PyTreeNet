"""
This module tests the direct application of a TTNO to a TTNS.
"""
from __future__ import annotations
import unittest
from copy import deepcopy

import numpy.testing as npt

from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.ttns.ttns_ttno.direct_application import direct

class TestDirectApplicationSmallTTNS(unittest.TestCase):
    """
    Tests the direct application of a TTNO to a small TTNS.
    """
    def test_direct_application(self):
        """
        Tests the direct application of a TTNO to a small TTNS.
        """
        ttns, ttno = small_ttns_and_ttno()
        ref_ttns = deepcopy(ttns)
        ref_ttno = deepcopy(ttno)
        result = direct(ttns, ttno)
        # Reference tensors
        state, order_state = ref_ttns.to_vector()
        op, order_op = ref_ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        npt.assert_array_almost_equal(found, ref)

class TestDirectApplicationBigTTNS(unittest.TestCase):
    """
    Tests the direct application of a TTNO to a big TTNS.
    """
    def test_direct_application(self):
        """
        Tests the direct application of a TTNO to a big TTNS.
        """
        ttns, ttno = big_ttns_and_ttno()
        ref_ttns = deepcopy(ttns)
        ref_ttno = deepcopy(ttno)
        result = direct(ttns, ttno)
        # Reference tensors
        state, order_state = ref_ttns.to_vector()
        op, order_op = ref_ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        npt.assert_array_almost_equal(found, ref)

if __name__ == "__main__":
    unittest.main()