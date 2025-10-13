"""
This module tests the direct application of a TTNO to a TTNS.
"""
from __future__ import annotations
import unittest
from copy import deepcopy

import numpy as np
import numpy.testing as npt

from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.random.random_special_ttns import (random_binary_state,
                                                  random_ftps,
                                                  random_mps,
                                                  random_tstar_state)
from pytreenet.random.random_special_ttno import (random_binary_operator,
                                                  random_ftpo,
                                                  random_mpo,
                                                  random_tstar_operator)
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

class TestDirectApplicationForSpecialTTNS(unittest.TestCase):
    """
    Tests the direct application of a TTNO to a TTNS for some special tree
    structures.
    """

    def test_for_mps(self):
        """
        Tests the direct application of an MPO to an MPS.
        """
        mps = random_mps(num_sites=10, bond_dim=20, phys_dim=2, seed=55586)
        mpo = random_mpo(num_sites=10, bond_dim=15, phys_dim=2, seed=55586)
        ref_mps = deepcopy(mps)
        ref_mpo = deepcopy(mpo)
        result = direct(mps, mpo)
        # Reference tensors
        state, order_state = ref_mps.to_vector()
        op, order_op = ref_mpo.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        found = found / (found.T.conj()@found)
        ref = ref / (ref.T.conj()@ref)
        npt.assert_array_almost_equal(found, ref)

    def test_for_binary(self):
        """
        Test the TTNO application for a binary tree structure.
        """
        ttns = random_binary_state(4,2,20,seed=39383)
        ttno = random_binary_operator(4,2,10,seed=39383)
        ref_ttns = deepcopy(ttns)
        ref_ttno = deepcopy(ttno)
        result = direct(ttns,ttno)
        # Reference tensors
        state, order_state = ref_ttns.to_vector()
        op, order_op = ref_ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        found = found / (found.T.conj()@found)
        ref = ref / (ref.T.conj()@ref)
        npt.assert_array_almost_equal(found, ref)

    def test_for_tstar(self):
        """
        Test the TTNO application for a tstar tree structure.
        """
        ttns = random_tstar_state(4,2,20,seed=39383)
        ttno = random_tstar_operator(4,2,10,seed=39383)
        ref_ttns = deepcopy(ttns)
        ref_ttno = deepcopy(ttno)
        result = direct(ttns,ttno)
        # Reference tensors
        state, order_state = ref_ttns.to_vector()
        op, order_op = ref_ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        found = found / (found.T.conj()@found)
        ref = ref / (ref.T.conj()@ref)
        npt.assert_array_almost_equal(found, ref)

    def test_for_ftps(self):
        """
        Test the TTNO application for an ftps tree structure.
        """
        ttns = random_ftps(3,2,20,seed=39383)
        ttno = random_ftpo(3,2,10,seed=39383)
        ref_ttns = deepcopy(ttns)
        ref_ttno = deepcopy(ttno)
        result = direct(ttns,ttno)
        # Reference tensors
        state, order_state = ref_ttns.to_vector()
        op, order_op = ref_ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        found = found / (found.T.conj()@found)
        ref = ref / (ref.T.conj()@ref)
        npt.assert_array_almost_equal(found, ref)

if __name__ == "__main__":
    unittest.main()
