"""
This module implements unittest for contracting two Tree Tensor Network states.
"""
import unittest

import numpy.testing as npt

from pytreenet.contractions.state_state_contraction import (contract_two_ttns,
                                                            scalar_product)
from pytreenet.random.random_ttns import (random_small_ttns,
                                          random_big_ttns,
                                          random_big_ttns_two_root_children)
from pytreenet.random.random_special_ttns import (random_binary_state,
                                                  random_ftps,
                                                  random_mps,
                                                  random_tstar_state)

class TestContractTwoTTNS(unittest.TestCase):
    """
    Tests for the contraction of two Tree Tensor Network states.
    """

    def setUp(self):
        self.seed = 684

    def test_small_ttns(self):
        """
        Tests contraction of small TTNs with known results.
        """
        ttn1 = random_small_ttns(seed=self.seed)
        ttn2 = random_small_ttns(seed=self.seed+1)
        result = contract_two_ttns(ttn1, ttn2)
        # Reference
        vec1, _ = ttn1.to_vector()
        vec2, _ = ttn2.to_vector()
        ref = vec1.T @ vec2
        npt.assert_allclose(result, ref)

    def test_big_ttns(self):
        """
        Tests contraction of bigger TTNs with known results.
        """
        ttn1 = random_big_ttns(seed=self.seed)
        ttn2 = random_big_ttns(seed=self.seed+1)
        result = contract_two_ttns(ttn1, ttn2)
        # Reference
        vec1, _ = ttn1.to_vector()
        vec2, _ = ttn2.to_vector()
        ref = vec1.T @ vec2
        npt.assert_allclose(result, ref)

    def test_big_ttns_two_root_children(self):
        """
        Tests contraction of bigger TTNs with known results.
        """
        ttn1 = random_big_ttns_two_root_children(seed=self.seed)
        ttn2 = random_big_ttns_two_root_children(seed=self.seed+1)
        result = contract_two_ttns(ttn1, ttn2)
        # Reference
        vec1, _ = ttn1.to_vector()
        vec2, _ = ttn2.to_vector()
        ref = vec1.T @ vec2
        npt.assert_allclose(result, ref)

    def test_random_mps(self):
        """
        Tests contraction of random MPS with known results.
        """
        mps1 = random_mps(10, 3, 14, seed=self.seed)
        mps2 = random_mps(10, 3, 10, seed=self.seed+1)
        result = contract_two_ttns(mps1, mps2)
        # Reference
        vec1, _ = mps1.to_vector()
        vec2, _ = mps2.to_vector()
        ref = vec1.T @ vec2
        npt.assert_allclose(result, ref)

    def test_random_binary_state(self):
        """
        Tests contraction of random binary tree states with known results.
        """
        bin1 = random_binary_state(4, 2, 20, seed=self.seed)
        bin2 = random_binary_state(4, 2, 15, seed=self.seed+1)
        result = contract_two_ttns(bin1, bin2)
        # Reference
        vec1, _ = bin1.to_vector()
        vec2, _ = bin2.to_vector()
        ref = vec1.T @ vec2
        npt.assert_allclose(result, ref)

    def test_random_tstar_state(self):
        """
        Tests contraction of random T* states with known results.
        """
        tstar1 = random_tstar_state(3, 3, 15, seed=self.seed)
        tstar2 = random_tstar_state(3, 3, 11, seed=self.seed+1)
        result = contract_two_ttns(tstar1, tstar2)
        # Reference
        vec1, _ = tstar1.to_vector()
        vec2, _ = tstar2.to_vector()
        ref = vec1.T @ vec2
        npt.assert_allclose(result, ref)

    def test_random_ftps(self):
        """
        Tests contraction of random FTPS with known results.
        """
        ftps1 = random_ftps(3, 2, 13, seed=self.seed)
        ftps2 = random_ftps(3, 2, 10, seed=self.seed+1)
        result = contract_two_ttns(ftps1, ftps2)
        # Reference
        vec1, _ = ftps1.to_vector()
        vec2, _ = ftps2.to_vector()
        ref = vec1.T @ vec2
        npt.assert_allclose(result, ref)

class TestScalarProduct(unittest.TestCase):
    """
    Tests for the scalar product of two Tree Tensor Network states.
    """

    def setUp(self):
        self.seed = 684


    def test_small_ttns(self):
        """
        Tests scalar product of small TTNs with known results.
        """
        ttn1 = random_small_ttns(seed=self.seed)
        result = scalar_product(ttn1)
        # Reference
        vec1, _ = ttn1.to_vector()
        ref = vec1.T.conj() @ vec1
        npt.assert_allclose(result, ref)

    def test_big_ttns(self):
        """
        Tests scalar product of bigger TTNs with known results.
        """
        ttn1 = random_big_ttns(seed=self.seed)
        result = scalar_product(ttn1)
        # Reference
        vec1, _ = ttn1.to_vector()
        ref = vec1.T.conj() @ vec1
        npt.assert_allclose(result, ref)

    def test_big_ttns_two_root_children(self):
        """
        Tests scalar product of bigger TTNs with known results.
        """
        ttn1 = random_big_ttns_two_root_children(seed=self.seed)
        result = scalar_product(ttn1)
        # Reference
        vec1, _ = ttn1.to_vector()
        ref = vec1.T.conj() @ vec1
        npt.assert_allclose(result, ref)

    def test_random_mps(self):
        """
        Tests scalar product of random MPS with known results.
        """
        mps1 = random_mps(10, 3, 14, seed=self.seed)
        result = scalar_product(mps1)
        # Reference
        vec1, _ = mps1.to_vector()
        ref = vec1.T.conj() @ vec1
        npt.assert_allclose(result, ref)

    def test_random_binary_state(self):
        """
        Tests scalar product of random binary tree states with known results.
        """
        bin1 = random_binary_state(4, 2, 20, seed=self.seed)
        result = scalar_product(bin1)
        # Reference
        vec1, _ = bin1.to_vector()
        ref = vec1.T.conj() @ vec1
        npt.assert_allclose(result, ref)
