import unittest

from copy import deepcopy

import numpy as np
from scipy.linalg import expm

import pytreenet as ptn
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.random.random_mps_and_mpo import random_mps_and_mpo
from pytreenet.contractions.zipup import zipup

class TestVariationalFittingSmall(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        ttns, self.ttno = small_ttns_and_ttno()
        ttns.canonical_form(ttns.root_id)
        ttns.normalize()
        self.ttns = ttns
        self.num_sweeps = 10
        self.max_iter = 50

        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(float("inf"), float("-inf"), float("-inf"))

        state_y = deepcopy(self.ttns)
        state_y.pad_bond_dimensions(6)
        self.varfit_one_site = ptn.VariationalFitting([self.ttno], [self.ttns], state_y, self.num_sweeps,
                             self.max_iter, self.svd_params, "one-site")

    def test_one_site_als(self):
        state_num = zipup(self.ttno, self.ttns)
        state_num.normalize()
        
        es=self.varfit_one_site.run()
        state_y = self.varfit_one_site.y
        state_y.normalize()
        self.assertAlmostEqual(state_y.scalar_product(state_num), 1, places=4)
        
class TestVariationalFittingBig(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        ttns, self.ttno = big_ttns_and_ttno()
        ttns.canonical_form(ttns.root_id)
        ttns.normalize()
        self.ttns = ttns
        self.num_sweeps = 10
        self.max_iter = 50

        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(float("inf"), float("-inf"), float("-inf"))

        state_y = deepcopy(self.ttns)
        state_y.pad_bond_dimensions(6)
        self.varfit_one_site = ptn.VariationalFitting([self.ttno], [deepcopy(self.ttns)], state_y, self.num_sweeps,
                             self.max_iter, self.svd_params, "one-site")

    def test_one_site_als(self):
        state_num = zipup(self.ttno, self.ttns,self.svd_params)
        state_num.normalize()
        
        es=self.varfit_one_site.run()
        state_y = self.varfit_one_site.y
        state_y.normalize()
        self.assertAlmostEqual(state_y.scalar_product(state_num), 1, places=1)
        
class TestVariationalFittingMPS(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        self.ttns, self.ttno = random_mps_and_mpo(2, 3)
        
        self.num_sweeps = 10
        self.max_iter = 50

        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(float("inf"), float("-inf"), float("-inf"))

        state_y = deepcopy(self.ttns)
        state_y.pad_bond_dimensions(6)
        self.varfit_one_site = ptn.VariationalFitting([self.ttno], [deepcopy(self.ttns)], state_y, self.num_sweeps,
                             self.max_iter, self.svd_params, "one-site")

    def test_one_site_als(self):
        state_num = zipup(self.ttno, self.ttns,self.svd_params)
        state_num.normalize()
        
        es=self.varfit_one_site.run()
        state_y = self.varfit_one_site.y
        state_y.normalize()
        self.assertAlmostEqual(state_y.scalar_product(state_num), 1, places=1)

if __name__ == "__main__":
    unittest.main()