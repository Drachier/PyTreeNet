"""
This module provides unittests to test the density matrix truncation method
"""
from __future__ import annotations
import unittest
from copy import deepcopy

import numpy.testing as npt

from pytreenet.core.truncation.density_matrix import density_matrix_truncation
from pytreenet.core.truncation.svd_truncation import svd_truncation, SVDParameters
from pytreenet.random.random_ttns import (random_small_ttns,
                                          random_big_ttns,
                                          RandomTTNSMode)

class TestDMFittingSmall(unittest.TestCase):
    """
    Tests the DM fitting on a small TTNS.
    """

    def test_dm_fitting_same_bd(self):
        """
        Tests the DM fitting on a small TTNS where the ansatz has the
        same bond dimension as the target.
        """
        init_ttns = random_small_ttns()
        init_ttns.normalise()
        ref_ttns = deepcopy(init_ttns)
        init_ttns = density_matrix_truncation(init_ttns, SVDParameters())
        fit_vec, _ = init_ttns.completely_contract_tree()
        corr_vec, _ = ref_ttns.completely_contract_tree()
        npt.assert_allclose(corr_vec, fit_vec, atol=1e-6)

    def test_var_fitting_lower_bd(self):
        """
        Tests the DM fitting on a small TTNS where the ansatz has a
        lower bond dimension than the target.
        """
        ttns = random_small_ttns()
        ttns = svd_truncation(ttns, SVDParameters(max_bond_dim=2))
        ttns.pad_bond_dimensions(4) # TTNS has a too high bd
        compressed = density_matrix_truncation(ttns,
                                               SVDParameters(max_bond_dim=2))
        found_vec, _ = compressed.completely_contract_tree()
        corr_vec, _ = compressed.completely_contract_tree()
        npt.assert_allclose(corr_vec, found_vec)

class TestDMFittingBig(unittest.TestCase):
    """
    Tests the DM fitting on a bigger TTNS.
    """

    def test_dm_fitting_same_bd(self):
        """
        Tests the DM fitting on a big TTNS where the ansatz has the
        same bond dimension as the target.
        """
        init_ttns = random_big_ttns()
        init_ttns.normalise()
        ref_ttns = deepcopy(init_ttns)
        init_ttns = density_matrix_truncation(init_ttns, SVDParameters())
        fit_vec, _ = init_ttns.completely_contract_tree()
        corr_vec, _ = ref_ttns.completely_contract_tree()
        npt.assert_allclose(corr_vec, fit_vec, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
