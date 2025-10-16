"""
This module test the zip-up method for applying a TTNO to a TTNS.
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

from pytreenet.ttns.ttns_ttno.zipup import zipup

class TestDMApproachrandomTTN(unittest.TestCase):
    """
    Test the DM apporach for a random TTNs.
    """
    def setUp(self):
        self.svd_params = SVDParameters(float("inf"), float("-inf"), float("-inf"))

    def test_small_ttn(self):
        """
        Test the DM approach for a small TTN.
        """
        ttns, ttno = small_ttns_and_ttno()
        result = zipup(ttno,
                        ttns,
                        svd_params=self.svd_params)
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        npt.assert_array_almost_equal(found, ref)

    def test_big_ttn(self):
        """
        Test the DM approach for a bigger TTN.
        """
        ttns, ttno = big_ttns_and_ttno()
        result = zipup(ttno,
                    ttns,
                    svd_params=self.svd_params)
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        npt.assert_array_almost_equal(found, ref)

class TestDMApproachSpecialTTN(unittest.TestCase):
    """
    Test the DM approach for special TTNs.
    """
    def setUp(self):
        self.svd_params = SVDParameters(float("inf"), float("-inf"), float("-inf"))

    def test_mps(self):
        """
        Test the DM approach for an MPS.
        """
        ttns = random_mps(7, 3, 20, seed=42)
        ttno = random_mpo(7, 3, 15, seed=24)
        result = zipup(ttno,
                       ttns,
                       svd_params=self.svd_params)
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
        Test the DM approach for a binary TTN state.
        """
        ttns = random_binary_state(3, 2, 10, seed=42)
        ttno = random_binary_operator(3, 2, 7, seed=24)
        result = zipup(ttno,
                       ttns,
                       svd_params=self.svd_params)
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
        Test the DM approach for a T* state.
        """
        ttns = random_tstar_state(3, 2, 15, seed=42)
        ttno = random_tstar_operator(3, 2, 10, seed=24)
        result = zipup(ttno,
                       ttns,
                       svd_params=self.svd_params)
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
        Test the DM approach for a FTPS.
        """
        ttns = random_ftps(3, 2, 10, seed=42)
        ttno = random_ftpo(3, 2, 7, seed=24)
        result = zipup(ttno,
                       ttns,
                       svd_params=self.svd_params)
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        found = found / (found.T.conj()@found)
        ref = ref / (ref.T.conj()@ref)
        npt.assert_array_almost_equal(found, ref)


if __name__ == "__main__":
    unittest.main()
        