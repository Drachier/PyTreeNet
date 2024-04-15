import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn

class TestTwoSiteTDVPSimple(unittest.TestCase):
    def setUp(self) -> None:
        self.ttn = ptn.random_small_ttns(ptn.RandomTTNSMode.DIFFVIRT)
        self.time_step_size = 0.1
        self.final_time = 1
        self.hamiltonian = ptn.random_hermitian_matrix((2*3*4))
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
        new_cache = ptn.contract_any("c1", "root",
                                     self.tdvp.state, self.tdvp.hamiltonian,
                                     self.tdvp.partial_tree_cache)
        self.tdvp.partial_tree_cache.add_entry("c1", "root", new_cache)
        new_cache = ptn.contract_any("root", "c2",
                                     self.tdvp.state, self.tdvp.hamiltonian,
                                     self.tdvp.partial_tree_cache)
        self.tdvp.partial_tree_cache.add_entry("root", "c2", new_cache)

    def test_find_block_leg_target_node_rc1(self):
        """
        Test the function find_block_leg_target_node if the target node is root
         and the next node is c1. The only neighbour available is c2.
        """
        target_node_id = "root"
        next_node_id = "c1"
        neighbour_id = "c2"
        found = self.tdvp._find_block_leg_target_node(target_node_id,
                                                      next_node_id,
                                                      neighbour_id)
        correct = 2
        self.assertEqual(found,correct)

    def test_find_block_leg_target_node_rc2(self):
        """
        Test the function find_block_leg_target_node if the target node is root
         and the next node is c2. The only neighbour available is c1.
        """
        target_node_id = "root"
        next_node_id = "c2"
        neighbour_id = "c1"
        found = self.tdvp._find_block_leg_target_node(target_node_id,
                                                      next_node_id,
                                                      neighbour_id)
        correct = 2
        self.assertEqual(found,correct)

    def test_find_block_leg_next_node_rc1(self):
        """
        Test the function find_block_leg_next_node if the target node is root
         and the next node is c1. The only neighbour available is c2.
        """
        target_node_id = "c1"
        next_node_id = "root"
        neighbour_id = "c2"
        found = self.tdvp._find_block_leg_next_node(target_node_id,
                                                      next_node_id,
                                                      neighbour_id)
        correct = 4
        self.assertEqual(found,correct)

    def test_find_block_leg_next_node_rc2(self):
        """
        Test the function find_block_leg_next_node if the target node is root
         and the next node is c2. The only neighbour available is c1.
        """
        target_node_id = "c2"
        next_node_id = "root"
        neighbour_id = "c1"
        found = self.tdvp._find_block_leg_next_node(target_node_id,
                                                      next_node_id,
                                                      neighbour_id)
        correct = 4
        self.assertEqual(found,correct)

    def test_determine_two_site_leg_permutation_rc1(self):
        """
        Test the function _determine_two_site_leg_permutation if the target node is root
         and the next node is c1. 
        """
        target_node_id = "root"
        next_node_id = "c1"
        self.tdvp.state.contract_nodes(target_node_id,next_node_id,
                                       self.tdvp.create_two_site_id(target_node_id,
                                                                    next_node_id))
        found = self.tdvp._determine_two_site_leg_permutation(target_node_id,
                                                               next_node_id)
        correct = (3,0,4,2,1,5)
        self.assertEqual(found,correct)

    def test_determine_two_site_leg_permutation_rc2(self):
        """
        Test the function _determine_two_site_leg_permutation if the target node is root
         and the next node is c2. 
        """
        target_node_id = "root"
        next_node_id = "c2"
        self.tdvp.state.contract_nodes(target_node_id,next_node_id,
                                       self.tdvp.create_two_site_id(target_node_id,
                                                                    next_node_id))
        found = self.tdvp._determine_two_site_leg_permutation(target_node_id,
                                                               next_node_id)
        correct = (3,0,4,2,1,5)
        self.assertEqual(found,correct)

    def test_determine_two_site_leg_permutation_c1r(self):
        """
        Test the function _determine_two_site_leg_permutation if the target node is c1
         and the next node is root. 
        """
        next_node_id = "root"
        target_node_id = "c1"
        self.tdvp.state.contract_nodes(target_node_id,next_node_id,
                                       self.tdvp.create_two_site_id(target_node_id,
                                                                    next_node_id))
        found = self.tdvp._determine_two_site_leg_permutation(target_node_id,
                                                               next_node_id)
        correct = (5,0,2,4,1,3)
        self.assertEqual(found,correct)

    def test_determine_two_site_leg_permutation_c2r(self):
        """
        Test the function _determine_two_site_leg_permutation if the target node is c2
         and the next node is root. 
        """
        next_node_id = "root"
        target_node_id = "c2"
        self.tdvp.state.contract_nodes(target_node_id,next_node_id,
                                       self.tdvp.create_two_site_id(target_node_id,
                                                                    next_node_id))
        found = self.tdvp._determine_two_site_leg_permutation(target_node_id,
                                                               next_node_id)
        correct = (5,0,2,4,1,3)
        self.assertEqual(found,correct)

    def test_contract_all_except_two_nodes_rc1(self):
        """
        Test the function contract_all_except_two_nodes if the target node is root
         and the next node is c1. We do a reference contraction to obtain a
         reference effective Hamiltonian.
        """
        target_node_id = "root"
        next_node_id = "c1"
        neighbour_id = "c2"
        # Reference contraction
        h_eff = np.tensordot(self.hamiltonian.tensors[target_node_id],
                             self.tdvp.partial_tree_cache.get_entry(neighbour_id,
                                                                    target_node_id),
                             axes=(1,1))
        h_eff = np.tensordot(h_eff,
                             self.hamiltonian.tensors[next_node_id],
                             axes=(0,0))
        h_eff = h_eff.transpose(3,0,4,2,1,5)
        self.tdvp.state.contract_nodes(target_node_id,next_node_id,
                                       self.tdvp.create_two_site_id(target_node_id,
                                                                    next_node_id))
        found = self.tdvp._contract_all_except_two_nodes(target_node_id,
                                                         next_node_id)
        self.assertTrue(np.allclose(found,h_eff))

    def test_contract_all_except_two_nodes_c1r(self):
        """
        Test the function contract_all_except_two_nodes if the target node is root
         and the next node is c1. We do a reference contraction to obtain a
         reference effective Hamiltonian. The result can be different to the above
         test, as the inputs are reversed.
        """
        target_node_id = "c1"
        next_node_id = "root"
        neighbour_id = "c2"
        # Reference contraction
        h_eff = np.tensordot(self.hamiltonian.tensors[next_node_id],
                             self.tdvp.partial_tree_cache.get_entry(neighbour_id,
                                                                    next_node_id),
                             axes=(1,1))
        h_eff = np.tensordot(h_eff,
                             self.hamiltonian.tensors[target_node_id],
                             axes=(0,0))
        h_eff = h_eff.transpose(3,4,0,2,5,1)
        self.tdvp.state.contract_nodes(target_node_id,next_node_id,
                                       self.tdvp.create_two_site_id(target_node_id,
                                                                    next_node_id))
        found = self.tdvp._contract_all_except_two_nodes(target_node_id,
                                                         next_node_id)
        self.assertTrue(np.allclose(found,h_eff))

    def test_get_effective_two_site_hamiltonian_rc1(self):
        """
        Tests the construction of the effective Hamiltonian if the target
         node is the root and the other node is c1.
        """
        target_node_id = "root"
        next_node_id = "c1"
        neighbour_id = "c2"
        # Reference contraction
        h_eff = np.tensordot(self.hamiltonian.tensors[target_node_id],
                             self.tdvp.partial_tree_cache.get_entry(neighbour_id,
                                                                    target_node_id),
                             axes=(1,1))
        h_eff = np.tensordot(h_eff,
                             self.hamiltonian.tensors[next_node_id],
                             axes=(0,0))
        h_eff = h_eff.transpose(3,0,4,2,1,5)
        h_eff = h_eff.reshape(36,36)
        self.tdvp.state.contract_nodes(target_node_id,next_node_id,
                                       self.tdvp.create_two_site_id(target_node_id,
                                                                    next_node_id))
        found = self.tdvp._get_effective_two_site_hamiltonian(target_node_id,
                                                              next_node_id)
        self.assertTrue(np.allclose(found,h_eff))

if __name__ == '__main__':
    unittest.main()
