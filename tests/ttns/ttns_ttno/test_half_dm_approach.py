"""
This module provides unittests for the density matrix based TTNO application.
"""
import unittest

import numpy.testing as npt

from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.random.random_special_ttns import (random_ftps,
                                                  random_binary_state,
                                                  random_mps,
                                                  random_tstar_state)
from pytreenet.random.random_special_ttno import (random_ftpo,
                                                  random_binary_operator,
                                                  random_mpo,
                                                  random_tstar_operator)
from pytreenet.util.tensor_splitting import SVDParameters

from pytreenet.ttns.ttns_ttno.half_dm_approach import half_dm_ttns_ttno_application

class TestHalfDMApproachrandomTTN(unittest.TestCase):
    """
    Test the half DM apporach for a random TTNs.
    """

    def test_small_ttn(self):
        """
        Test the half DM approach for a small TTN.
        """
        ttns, ttno = small_ttns_and_ttno()
        result = half_dm_ttns_ttno_application(ttns,
                                                ttno,
                                                svd_params=SVDParameters(max_bond_dim=10))
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        npt.assert_array_almost_equal(found, ref)

    def test_big_ttn(self):
        """
        Test the half DM approach for a bigger TTN.
        """
        ttns, ttno = big_ttns_and_ttno()
        result = half_dm_ttns_ttno_application(ttns,
                                                ttno,
                                                svd_params=SVDParameters(max_bond_dim=20))
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        npt.assert_array_almost_equal(found, ref)

class TestHalfDMApproachSpecialTTN(unittest.TestCase):
    """
    Test the half DM approach for special TTNs.
    """

    def test_mps(self):
        """
        Test the half DM approach for an MPS.
        """
        ttns = random_mps(7, 3, 20, seed=42)
        ttno = random_mpo(7, 3, 15, seed=24)
        result = half_dm_ttns_ttno_application(ttns,
                                                ttno,
                                                svd_params=SVDParameters(max_bond_dim=100))
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        found = found / (found.T.conj()@found)
        ref = ref / (ref.T.conj()@ref)
        npt.assert_array_almost_equal(found, ref)

    def test_binary_state(self):
        """
        Test the half DM approach for a binary TTN state.
        """
        ttns = random_binary_state(3, 2, 10, seed=42)
        ttno = random_binary_operator(3, 2, 7, seed=24)
        result = half_dm_ttns_ttno_application(ttns,
                                                ttno,
                                                svd_params=SVDParameters(max_bond_dim=100))
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        found = found / (found.T.conj()@found)
        ref = ref / (ref.T.conj()@ref)
        npt.assert_array_almost_equal(found, ref)

    def test_tstar_state(self):
        """
        Test the half DM approach for a T* state.
        """
        ttns = random_tstar_state(3, 2, 15, seed=42)
        ttno = random_tstar_operator(3, 2, 10, seed=24)
        result = half_dm_ttns_ttno_application(ttns,
                                                ttno,
                                                svd_params=SVDParameters(max_bond_dim=100))
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        found = found / (found.T.conj()@found)
        ref = ref / (ref.T.conj()@ref)
        npt.assert_array_almost_equal(found, ref)

    def test_ftps(self):
        """
        Test the half DM approach for a FTPS.
        """
        ttns = random_ftps(3, 2, 10, seed=42)
        ttno = random_ftpo(3, 2, 7, seed=24)
        result = half_dm_ttns_ttno_application(ttns,
                                                ttno,
                                                svd_params=SVDParameters(max_bond_dim=100))
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        found = found / (found.T.conj()@found)
        ref = ref / (ref.T.conj()@ref)
        npt.assert_array_almost_equal(found, ref)
