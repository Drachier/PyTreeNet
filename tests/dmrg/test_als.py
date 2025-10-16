import unittest

from copy import deepcopy

import numpy as np
from scipy.linalg import expm

import pytreenet as ptn
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.random.random_mps_and_mpo import random_mps_and_mpo
from pytreenet.ttns.ttns_ttno.zipup import zipup
np.random.seed(42)
class TestALSsmall(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        self.ttns, self.ttno = small_ttns_and_ttno()

        self.num_sweeps = 5
        self.max_iter = 50

        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(10, float("-inf"), float("-inf"))

        state_x = deepcopy(self.ttns)
        state_x.pad_bond_dimensions(6)
        self.als_one_site = ptn.AlternatingLeastSquares(self.ttno, state_x, deepcopy(self.ttns), self.num_sweeps,
                             self.max_iter, self.svd_params, "one-site",1,dtype=np.complex128)

    def test_one_site_als(self):
        state_old = zipup(self.ttno, self.ttns)
        state_old.normalize()
        old_olvp = state_old.scalar_product(self.ttns)
        
        es=self.als_one_site.run()
        state_x = self.als_one_site.state_x
        state_b_num = zipup(self.ttno, state_x)
        new_olvp = state_b_num.scalar_product(self.ttns)/state_b_num.scalar_product(state_b_num)
        
        self.assertGreater(new_olvp.real, old_olvp.real)
        
class TestALSMPS(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        self.ttns, self.ttno = random_mps_and_mpo(2, 3)
        
        self.num_sweeps = 5
        self.max_iter = 50

        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(10, float("-inf"), float("-inf"))

        state_x = deepcopy(self.ttns)
        state_x.pad_bond_dimensions(6)
        self.als_one_site = ptn.AlternatingLeastSquares(self.ttno, state_x, self.ttns, self.num_sweeps,
                             self.max_iter, self.svd_params, "one-site",1,dtype=np.complex128)

        
    def test_one_site_als(self):
        state_old = zipup(self.ttno, self.ttns)
        state_old.normalize()
        old_olvp = state_old.scalar_product(self.ttns)
        
        es=self.als_one_site.run()
        state_x = self.als_one_site.state_x
        state_b_num = zipup(self.ttno, state_x)
        new_olvp = state_b_num.scalar_product(self.ttns)/state_b_num.scalar_product(state_b_num)
        
        self.assertGreater(new_olvp.real, old_olvp.real)

        
class TestALSBig(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        ttns, self.ttno = big_ttns_and_ttno()
        ttns.canonical_form(ttns.root_id)
        ttns.normalize()
        self.ttns = ttns
        self.num_sweeps = 5
        self.max_iter = 50

        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(10, float("-inf"), float("-inf"))

        state_x = deepcopy(self.ttns)
        state_x.pad_bond_dimensions(6)
        self.als_one_site = ptn.AlternatingLeastSquares(self.ttno, state_x, deepcopy(self.ttns), self.num_sweeps,
                             self.max_iter, self.svd_params, "one-site",1,dtype=np.complex128)

    def test_one_site_als(self):        
        state_old = zipup(self.ttno, self.ttns,self.svd_params)
        state_old.normalize()
        old_olvp = state_old.scalar_product(self.ttns)
        
        es=self.als_one_site.run()
        state_x = self.als_one_site.state_x
        state_b_num = zipup(self.ttno, state_x)
        new_olvp = state_b_num.scalar_product(self.ttns)/state_b_num.scalar_product(state_b_num)
        
        self.assertGreater(new_olvp.real, old_olvp.real)
        
if __name__ == "__main__":
    unittest.main()
