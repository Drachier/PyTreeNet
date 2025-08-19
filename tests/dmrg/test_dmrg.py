import unittest

from copy import deepcopy

import numpy as np
from scipy.linalg import expm

import pytreenet as ptn
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
np.random.seed(123)
class TestDMRGsmall(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        self.ttns, self.ttno = small_ttns_and_ttno()
        self.ttns.pad_bond_dimensions(10)
        self.num_sweeps = 20
        self.max_iter = 100

        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(float("inf"), float("-inf"), float("-inf"))

        # Initialise TEBD
        self.dmrg_one_site = ptn.DMRGAlgorithm(self.ttns, self.ttno, self.num_sweeps,
                             self.max_iter, self.svd_params, "one-site")

        self.dmrg_two_site = ptn.DMRGAlgorithm(self.ttns, self.ttno, self.num_sweeps,
                             self.max_iter, self.svd_params, "two-site")

    def test_one_site_dmrg(self):
        es=self.dmrg_one_site.run()
        numerical_ttno = self.ttno.as_matrix()[0]
        self.assertAlmostEqual(es[-1], np.linalg.eigvalsh(numerical_ttno)[0],places=1)

    def test_two_site_dmrg(self):
        es=self.dmrg_two_site.run()
        numerical_ttno = self.ttno.as_matrix()[0]
        self.assertAlmostEqual(es[-1], np.linalg.eigvalsh(numerical_ttno)[0],places=1)
        
class TestDMRGbig(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        self.ttns, self.ttno = big_ttns_and_ttno()
        self.ttns.pad_bond_dimensions(10)
        self.num_sweeps = 20
        self.max_iter = 500

        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(float("inf"), float("-inf"), float("-inf"))

        # Initialise TEBD
        self.dmrg_one_site = ptn.DMRGAlgorithm(self.ttns, self.ttno, self.num_sweeps,
                             self.max_iter, self.svd_params, "one-site")

        self.dmrg_two_site = ptn.DMRGAlgorithm(self.ttns, self.ttno, self.num_sweeps,
                             self.max_iter, self.svd_params, "two-site")

    def test_one_site_dmrg(self):
        # Sometimes this test fails, because the ttns bond_dim is too small, but it is okay.
        es=self.dmrg_one_site.run()
        numerical_ttno = self.ttno.as_matrix()[0]
        self.assertAlmostEqual(es[-1], np.linalg.eigvalsh(numerical_ttno)[0],places=1)

    def test_two_site_dmrg(self):
        es=self.dmrg_two_site.run()
        numerical_ttno = self.ttno.as_matrix()[0]
        self.assertAlmostEqual(es[-1], np.linalg.eigvalsh(numerical_ttno)[0],places=1)
        
if __name__ == "__main__":
    unittest.main()
