import unittest

from copy import deepcopy

import numpy as np
from scipy.linalg import expm

import pytreenet as ptn
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
# np.random.seed(4234256543)
class TestDMRGsmall(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        self.ttns, self.ttno = small_ttns_and_ttno()
        self.ttns.canonical_form(self.ttns.root_id)
        self.ttns.normalise()
        self.num_sweeps = 10
        self.max_iter = 100

        # Deactivate Truncation
        # self.svd_params = ptn.SVDParameters(float("inf"), float("-inf"), float("-inf"))
        self.svd_params = ptn.SVDParameters(5, 1e-10, 1e-10)

    def test_one_site_dmrg(self):
        precond = lambda state, svd_params: ptn.precond_lobpcg(self.ttno, state, svd_params)
        state_x, energy = ptn.lobpcg_single(self.ttno, self.ttns, precond, self.svd_params, self.num_sweeps, [])

        numerical_ttno = self.ttno.as_matrix()[0]
        
        numerical_state = state_x.completely_contract_tree()[0]
        numerical_state = numerical_state.flatten()
        
if __name__ == "__main__":
    unittest.main()