"""
This module provides unittests to test the variational truncation methods.
"""
from __future__ import annotations
import unittest

import numpy.testing as npt

from pytreenet.core.truncation.variational import single_site_fitting
from pytreenet.core.truncation.svd_truncation import svd_truncation, SVDParameters
from pytreenet.random.random_ttns import (random_small_ttns,
                                          random_big_ttns,
                                          RandomTTNSMode)

class TestVariationalFittingSmall(unittest.TestCase):
    """
    Tests the variational fitting on a small TTNS.
    """
    def setUp(self) -> None:
        self.ttns_dp = random_small_ttns()

    def test_var_fitting_same_bd(self):
        """
        Tests the variational fitting on a small TTNS where the ansatz has the
        same bond dimension as the target.
        """
        init_ttns = random_small_ttns()
        init_ttns.normalise()
        self.ttns_dp.normalise()
        _ = single_site_fitting(init_ttns, self.ttns_dp, num_sweeps=2,
                                    record_sweep_errors=True)
        corr_vec, _ = self.ttns_dp.completely_contract_tree()
        fit_vec, _ = init_ttns.completely_contract_tree()
        npt.assert_allclose(corr_vec, fit_vec, atol=1e-6)

    def test_var_fitting_lower_bd(self):
        """
        Tests the variational fitting on a small TTNS where the ansatz has a
        lower bond dimension than the target.
        """
        svd_truncation(self.ttns_dp, SVDParameters(max_bond_dim=2))
        self.ttns_dp.pad_bond_dimensions(4)
        init_ttns = random_small_ttns(mode=RandomTTNSMode.SAMEVIRT)
        init_ttns.normalise()
        self.ttns_dp.normalise()
        _ = single_site_fitting(init_ttns, self.ttns_dp, num_sweeps=5,
                                    record_sweep_errors=True)
        corr_vec, _ = self.ttns_dp.completely_contract_tree()
        fit_vec, _ = init_ttns.completely_contract_tree()
        fit_vec = fit_vec.transpose((0,2,1))
        npt.assert_allclose(corr_vec, fit_vec, atol=1e-6)

class TestVariationalFittingBig(unittest.TestCase):
    """
    Tests the variational fitting on a bigger TTNS.
    """

    def setUp(self) -> None:
        self.ttns_dp = random_big_ttns()

    def test_var_fitting_same_bd(self):
        """
        Tests the variational fitting on a big TTNS where the ansatz has the
        same bond dimension as the target.
        """
        init_ttns = random_big_ttns(mode=RandomTTNSMode.SAME)
        init_ttns.normalise()
        self.ttns_dp.normalise()
        _ = single_site_fitting(init_ttns, self.ttns_dp, num_sweeps=2,
                                    record_sweep_errors=True)
        corr_vec, _ = self.ttns_dp.completely_contract_tree()
        fit_vec, _ = init_ttns.completely_contract_tree()
        fit_vec = fit_vec.transpose((0,1,2,3,4,5,7,6))
        npt.assert_allclose(corr_vec, fit_vec, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
