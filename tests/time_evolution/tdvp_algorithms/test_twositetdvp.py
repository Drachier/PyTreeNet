import unittest

import numpy as np

import pytreenet as ptn
from pytreenet.random import (random_small_ttns,
                              RandomTTNSMode,
                              random_hermitian_matrix)
from pytreenet.contractions.state_operator_contraction import contract_any

class TestTwoSiteTDVPSimple(unittest.TestCase):
    def setUp(self) -> None:
        self.ttn = random_small_ttns(RandomTTNSMode.DIFFVIRT)
        self.time_step_size = 0.1
        self.final_time = 1
        self.hamiltonian = random_hermitian_matrix((2*3*4))
        self.hamiltonian = self.hamiltonian.reshape(4,3,2,4,3,2)
        leg_dict = {"c1": 1, "root": 2, "c2": 0}
        self.operators = []
        self.hamiltonian = ptn.TTNO.from_tensor(self.ttn,
                                                self.hamiltonian,
                                                leg_dict)
        self.svd_param = ptn.SVDParameters(max_bond_dim=4,
                                           rel_tol=1e-6,
                                           total_tol=1e-6)
        self.tdvp = ptn.TwoSiteTDVP(self.ttn,
                                    self.hamiltonian,
                                    self.time_step_size,
                                    self.final_time,
                                    self.operators,
                                    self.svd_param)
        new_cache = contract_any("c1", "root",
                                     self.tdvp.state, self.tdvp.hamiltonian,
                                     self.tdvp.partial_tree_cache)
        self.tdvp.partial_tree_cache.add_entry("c1", "root", new_cache)
        new_cache = contract_any("root", "c2",
                                     self.tdvp.state, self.tdvp.hamiltonian,
                                     self.tdvp.partial_tree_cache)
        self.tdvp.partial_tree_cache.add_entry("root", "c2", new_cache)

if __name__ == '__main__':
    unittest.main()
