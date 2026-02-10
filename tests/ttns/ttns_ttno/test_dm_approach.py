"""
This module provides unittests for the density matrix based TTNO application.
"""
import unittest
from copy import deepcopy

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

from pytreenet.ttns.ttns_ttno.dm_approach import (dm_ttns_ttno_application,
                                                  dm_linear_combination)

class TestDMApproachrandomTTN(unittest.TestCase):
    """
    Test the DM apporach for a random TTNs.
    """

    def test_small_ttn(self):
        """
        Test the DM approach for a small TTN.
        """
        ttns, ttno = small_ttns_and_ttno()
        result = dm_ttns_ttno_application(ttns,
                                          ttno,
                                          svd_params=SVDParameters(max_bond_dim=10))
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        npt.assert_array_almost_equal(found, ref)

    def test_linear_combination_small_ttn(self):
        """
        Test the DM approach for a linear combination of small TTNs.
        """
        ttns1, ttno1 = small_ttns_and_ttno()
        ttns2, ttno2 = small_ttns_and_ttno()
        ttns3, ttno3 = small_ttns_and_ttno()
        ref = [deepcopy(ttno1).as_matrix()[0] @ deepcopy(ttns1).to_vector()[0],
               deepcopy(ttno2).as_matrix()[0] @ deepcopy(ttns2).to_vector()[0],
               deepcopy(ttno3).as_matrix()[0] @ deepcopy(ttns3).to_vector()[0]]
        ref = sum(ref)
        found = dm_linear_combination([ttns1, ttns2, ttns3],
                                      [ttno1, ttno2, ttno3])
        found = found.to_vector()[0]
        npt.assert_array_almost_equal(found, ref)

    def test_big_ttn(self):
        """
        Test the DM approach for a bigger TTN.
        """
        ttns, ttno = big_ttns_and_ttno()
        result = dm_ttns_ttno_application(ttns,
                                          ttno,
                                          svd_params=SVDParameters(max_bond_dim=20))
        # Reference tensors
        state, order_state = ttns.to_vector()
        op, order_op = ttno.as_matrix()
        ref = op @ state
        found, order = result.to_vector()
        npt.assert_array_almost_equal(found, ref)

    def test_linear_combination_big_ttn(self):
        """
        Test the DM approach for a linear combination of bigger TTNs.
        """
        ttns1, ttno1 = big_ttns_and_ttno()
        ttns2, ttno2 = big_ttns_and_ttno()
        ttns3, ttno3 = big_ttns_and_ttno()
        ref = [deepcopy(ttno1).as_matrix()[0] @ deepcopy(ttns1).to_vector()[0],
               deepcopy(ttno2).as_matrix()[0] @ deepcopy(ttns2).to_vector()[0],
               deepcopy(ttno3).as_matrix()[0] @ deepcopy(ttns3).to_vector()[0]]
        ref = sum(ref)
        found = dm_linear_combination([ttns1, ttns2, ttns3],
                                      [ttno1, ttno2, ttno3])
        found = found.to_vector()[0]
        npt.assert_array_almost_equal(found, ref)

class TestDMApproachSpecialTTN(unittest.TestCase):
    """
    Test the DM approach for special TTNs.
    """

    def test_mps(self):
        """
        Test the DM approach for an MPS.
        """
        ttns = random_mps(7, 3, 20, seed=42)
        ttno = random_mpo(7, 3, 15, seed=24)
        result = dm_ttns_ttno_application(ttns,
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

    def test_linear_combination_mps(self):
        """
        Test the DM approach for a linear combination of MPS.
        """
        ttns1 = random_mps(7, 3, 20, seed=42)
        ttno1 = random_mpo(7, 3, 15, seed=24)
        ttns2 = random_mps(7, 3, 20, seed=43)
        ttno2 = random_mpo(7, 3, 15, seed=25)
        ttns3 = random_mps(7, 3, 20, seed=44)
        ttno3 = random_mpo(7, 3, 15, seed=26)
        ref = [deepcopy(ttno1).as_matrix()[0] @ deepcopy(ttns1).to_vector()[0],
               deepcopy(ttno2).as_matrix()[0] @ deepcopy(ttns2).to_vector()[0],
               deepcopy(ttno3).as_matrix()[0] @ deepcopy(ttns3).to_vector()[0]]
        ref = sum(ref)
        ref = ref / (ref.T.conj()@ref)
        found = dm_linear_combination([ttns1, ttns2, ttns3],
                                      [ttno1, ttno2, ttno3])
        found = found.to_vector()[0]
        found = found / (found.T.conj()@found)
        npt.assert_array_almost_equal(found, ref)

    def test_binary_state(self):
        """
        Test the DM approach for a binary TTN state.
        """
        ttns = random_binary_state(3, 2, 10, seed=42)
        ttno = random_binary_operator(3, 2, 7, seed=24)
        result = dm_ttns_ttno_application(ttns,
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

    def test_linear_combination_binary_state(self):
        """
        Test the DM approach for a linear combination of binary TTN states.
        """
        ttns1 = random_binary_state(3, 2, 10, seed=42)
        ttno1 = random_binary_operator(3, 2, 7, seed=24)
        ttns2 = random_binary_state(3, 2, 10, seed=43)
        ttno2 = random_binary_operator(3, 2, 7, seed=25)
        ttns3 = random_binary_state(3, 2, 10, seed=44)
        ttno3 = random_binary_operator(3, 2, 7, seed=26)
        ref = [deepcopy(ttno1).as_matrix()[0] @ deepcopy(ttns1).to_vector()[0],
               deepcopy(ttno2).as_matrix()[0] @ deepcopy(ttns2).to_vector()[0],
               deepcopy(ttno3).as_matrix()[0] @ deepcopy(ttns3).to_vector()[0]]
        ref = sum(ref)
        ref = ref / (ref.T.conj()@ref)
        found = dm_linear_combination([ttns1, ttns2, ttns3],
                                      [ttno1, ttno2, ttno3])
        found = found.to_vector()[0]
        found = found / (found.T.conj()@found)
        npt.assert_array_almost_equal(found, ref)

    def test_tstar_state(self):
        """
        Test the DM approach for a T* state.
        """
        ttns = random_tstar_state(3, 2, 15, seed=42)
        ttno = random_tstar_operator(3, 2, 10, seed=24)
        result = dm_ttns_ttno_application(ttns,
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

    def test_linear_combination_tstar_state(self):
        """
        Test the DM approach for a linear combination of T* states.
        """
        ttns1 = random_tstar_state(3, 2, 15, seed=42)
        ttno1 = random_tstar_operator(3, 2, 10, seed=24)
        ttns2 = random_tstar_state(3, 2, 15, seed=43)
        ttno2 = random_tstar_operator(3, 2, 10, seed=25)
        ttns3 = random_tstar_state(3, 2, 15, seed=44)
        ttno3 = random_tstar_operator(3, 2, 10, seed=26)
        ref = [deepcopy(ttno1).as_matrix()[0] @ deepcopy(ttns1).to_vector()[0],
               deepcopy(ttno2).as_matrix()[0] @ deepcopy(ttns2).to_vector()[0],
               deepcopy(ttno3).as_matrix()[0] @ deepcopy(ttns3).to_vector()[0]]
        ref = sum(ref)
        ref = ref / (ref.T.conj()@ref)
        found = dm_linear_combination([ttns1, ttns2, ttns3],
                                      [ttno1, ttno2, ttno3])
        found = found.to_vector()[0]
        found = found / (found.T.conj()@found)
        npt.assert_array_almost_equal(found, ref)

    def test_ftps(self):
        """
        Test the DM approach for a FTPS.
        """
        ttns = random_ftps(3, 2, 10, seed=42)
        ttno = random_ftpo(3, 2, 7, seed=24)
        result = dm_ttns_ttno_application(ttns,
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

    def test_linear_combination_ftps(self):
        """
        Test the DM approach for a linear combination of FTPS.
        """
        ttns1 = random_ftps(3, 2, 10, seed=42)
        ttno1 = random_ftpo(3, 2, 7, seed=24)
        ttns2 = random_ftps(3, 2, 10, seed=43)
        ttno2 = random_ftpo(3, 2, 7, seed=25)
        ttns3 = random_ftps(3, 2, 10, seed=44)
        ttno3 = random_ftpo(3, 2, 7, seed=26)
        ref = [deepcopy(ttno1).as_matrix()[0] @ deepcopy(ttns1).to_vector()[0],
               deepcopy(ttno2).as_matrix()[0] @ deepcopy(ttns2).to_vector()[0],
               deepcopy(ttno3).as_matrix()[0] @ deepcopy(ttns3).to_vector()[0]]
        ref = sum(ref)
        ref = ref / (ref.T.conj()@ref)
        found = dm_linear_combination([ttns1, ttns2, ttns3],
                                      [ttno1, ttno2, ttno3])
        found = found.to_vector()[0]
        found = found / (found.T.conj()@found)
        npt.assert_array_almost_equal(found, ref)
