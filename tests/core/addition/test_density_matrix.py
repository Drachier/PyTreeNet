"""
Module to test DM addition of Tree Tensor Networks (TTNs).
"""
import unittest
from copy import deepcopy

import numpy as np

from pytreenet.core.addition.density_matrix import (density_matrix_addition)
from pytreenet.random.random_ttns import (random_small_ttns,
                                          random_big_ttns)
from pytreenet.util.tensor_splitting import SVDParameters

class TestDensityMatrixForSmall(unittest.TestCase):
    """
    Test density matrix addition of small TTNs.
    """

    def test_double(self):
        """
        Test density matrix addition of small TTNs.
        """
        ttns1 = random_small_ttns(seed=42)
        ttns2 = deepcopy(ttns1)
        ref, _ = deepcopy(ttns1).completely_contract_tree()
        added_ttn = density_matrix_addition([ttns1, ttns2],
                                            SVDParameters())
        added, _ = added_ttn.completely_contract_tree()
        np.testing.assert_allclose(added, 2 * ref)

    def test_triple(self):
        """
        Test density matrix addition of small TTNs.
        """
        ttns1 = random_small_ttns(seed=42)
        ttns2 = deepcopy(ttns1)
        ttns3 = deepcopy(ttns1)
        ref, _ = deepcopy(ttns1).completely_contract_tree()
        added_ttn = density_matrix_addition([ttns1, ttns2, ttns3],
                                            SVDParameters())
        added, _ = added_ttn.completely_contract_tree()
        np.testing.assert_allclose(added, 3 * ref)

    def test_different(self):
        """
        Test density matrix addition of different small TTNs.
        """
        ttns1 = random_small_ttns(seed=42)
        ttns2 = random_small_ttns(seed=43)
        ref1, _ = deepcopy(ttns1).completely_contract_tree()
        ref2, _ = deepcopy(ttns2).completely_contract_tree()
        added_ttn = density_matrix_addition([ttns1, ttns2],
                                            SVDParameters())
        added, _ = added_ttn.completely_contract_tree()
        np.testing.assert_allclose(added, ref1 + ref2)

class TestDensityMatrixForBig(unittest.TestCase):
    """
    Test density matrix addition of big TTNs.
    """

    def test_double(self):
        """
        Test density matrix addition of big TTNs.
        """
        ttns1 = random_big_ttns(seed=52)
        ttns2 = deepcopy(ttns1)
        ref, _ = deepcopy(ttns1).completely_contract_tree()
        added_ttn = density_matrix_addition([ttns1, ttns2],
                                            SVDParameters())
        added, _ = added_ttn.completely_contract_tree()
        np.testing.assert_allclose(added, 2 * ref)

    def test_triple(self):
        """
        Test density matrix addition of big TTNs.
        """
        ttns1 = random_big_ttns(seed=52)
        ttns2 = deepcopy(ttns1)
        ttns3 = deepcopy(ttns1)
        ref, _ = deepcopy(ttns1).completely_contract_tree()
        added_ttn = density_matrix_addition([ttns1, ttns2, ttns3],
                                            SVDParameters())
        added, _ = added_ttn.completely_contract_tree()
        np.testing.assert_allclose(added, 3 * ref)

    def test_different(self):
        """
        Test density matrix addition of different big TTNs.
        """
        ttns1 = random_big_ttns(seed=52)
        ttns2 = random_big_ttns(seed=53)
        ref1, _ = deepcopy(ttns1).completely_contract_tree()
        ref2, _ = deepcopy(ttns2).completely_contract_tree()
        added_ttn = density_matrix_addition([ttns1, ttns2],
                                            SVDParameters())
        added, _ = added_ttn.completely_contract_tree()
        np.testing.assert_allclose(added, ref1 + ref2)

if __name__ == "__main__":
    unittest.main()
