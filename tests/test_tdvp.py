
import unittest
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

import pytreenet as ptn
from pytreenet.contractions.state_operator_contraction import (contract_any)

class TestTDVPInit(unittest.TestCase):

    def setUp(self):
        self.conversion_dict = {"root_op1": ptn.random_hermitian_matrix(),
                                "root_op2": ptn.random_hermitian_matrix(),
                                "I2": np.eye(2),
                                "c1_op": ptn.random_hermitian_matrix(size=3),
                                "I3": np.eye(3),
                                "c2_op": ptn.random_hermitian_matrix(size=4),
                                "I4": np.eye(4)}
        self.ref_tree = ptn.random_small_ttns()
        tensor_prod = [ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"})
                       ]
        ham = ptn.Hamiltonian(tensor_prod, self.conversion_dict)
        operator = ptn.TensorProduct({"root": ptn.crandn((2,2))})
        self.hamiltonian = ptn.TTNO.from_hamiltonian(ham, self.ref_tree)
        self.tdvp = ptn.TDVPAlgorithm(self.ref_tree, self.hamiltonian,
                                      0.1, 1, operator)

    def test_init_hamiltonian(self):
        self.assertEqual(self.hamiltonian, self.tdvp.hamiltonian)

    def test_state_initialisation(self):
        for node_id in self.ref_tree.nodes:
            ref_node, ref_tensor = self.ref_tree[node_id]
            node, tensor = self.tdvp.state[node_id]
            self.assertEqual(ref_node.shape, node.shape)
            self.assertEqual(ref_tensor.shape, tensor.shape)

    def test_init_update_path(self):
        self.assertEqual(["c1","root","c2"], self.tdvp.update_path)

    def test_init_orth_path(self):
        self.assertEqual([["root"],["c2"]], self.tdvp.orthogonalization_path)

    def test_find_caching_path(self):
        caching_path, next_node_id_dict = self.tdvp._find_caching_path()
        self.assertEqual(["c2", "root", "c1"], caching_path)
        self.assertEqual({"c2": "root", "root": "c1"}, next_node_id_dict)

    def test_init_partial_tree_cache(self):
        # Creating reference
        ref_tdvp = deepcopy(self.tdvp)
        partial_tree_cache = ptn.PartialTreeCachDict()
        c2_block = contract_any("c2", "root",
                                ref_tdvp.state,
                                ref_tdvp.hamiltonian,
                                partial_tree_cache)
        partial_tree_cache.add_entry("c2", "root", c2_block)
        root_block = contract_any("root", "c1",
                                  ref_tdvp.state,
                                  ref_tdvp.hamiltonian,
                                  partial_tree_cache)
        partial_tree_cache.add_entry("root", "c1", root_block)

        for ids, tensor in partial_tree_cache.items():
            found_tensor = self.tdvp.partial_tree_cache.get_entry(ids[0], ids[1])
            self.assertTrue(np.allclose(tensor, found_tensor))

class TestContractionMethods(unittest.TestCase):

    def setUp(self):
        self.conversion_dict = {"root_op1": ptn.random_hermitian_matrix(),
                                "root_op2": ptn.random_hermitian_matrix(),
                                "I2": np.eye(2),
                                "c1_op": ptn.random_hermitian_matrix(size=3),
                                "I3": np.eye(3),
                                "c2_op": ptn.random_hermitian_matrix(size=4),
                                "I4": np.eye(4)}
        self.ref_tree = ptn.random_small_ttns()
        tensor_prod = [ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"})
                       ]
        ham = ptn.Hamiltonian(tensor_prod, self.conversion_dict)
        operator = ptn.TensorProduct({"root": ptn.crandn((2,2))})
        self.hamiltonian = ptn.TTNO.from_hamiltonian(ham, self.ref_tree)
        self.tdvp = ptn.TDVPAlgorithm(self.ref_tree, self.hamiltonian,
                                      0.1, 1, operator)

        # Computing the other cached tensors for use
        c1_cache = contract_any("c1", "root",
                                self.tdvp.state, self.tdvp.hamiltonian,
                                self.tdvp.partial_tree_cache)
        self.tdvp.partial_tree_cache.add_entry("c1", "root", c1_cache)
        root_to_c2_cache = contract_any("root", "c2",
                                        self.tdvp.state, self.tdvp.hamiltonian,
                                        self.tdvp.partial_tree_cache)
        self.tdvp.partial_tree_cache.add_entry("root", "c2", root_to_c2_cache)

    def test_move_orth_and_update_cache_for_path_c1_to_c2(self):
        ref_tdvp = deepcopy(self.tdvp)
        ref_tdvp.state.move_orthogonalization_center("c2",
                                                     mode=ptn.SplitMode.KEEP)
        update_c1_cache = contract_any("c1", "root",
                                       ref_tdvp.state, ref_tdvp.hamiltonian,
                                       ref_tdvp.partial_tree_cache)
        ref_tdvp.partial_tree_cache.add_entry("c1","root",
                                              update_c1_cache)
        update_root_to_c2_cache = contract_any("root", "c2",
                                               ref_tdvp.state,
                                               ref_tdvp.hamiltonian,
                                               ref_tdvp.partial_tree_cache)
        ref_tdvp.partial_tree_cache.add_entry("root","c2",
                                              update_root_to_c2_cache)

    def test_move_orth_and_update_cache_for_path_c1_to_root_to_c2(self):
        path1 = ["c1","root"]
        ref_tdvp = deepcopy(self.tdvp)
        ref_tdvp.state.move_orthogonalization_center("root",
                                                     mode=ptn.SplitMode.KEEP)
        update_c1_cache = contract_any("c1", "root",
                                       ref_tdvp.state,
                                       ref_tdvp.hamiltonian,
                                       ref_tdvp.partial_tree_cache)
        ref_tdvp.partial_tree_cache.add_entry("c1","root",
                                              update_c1_cache)

        self.tdvp._move_orth_and_update_cache_for_path(path1)

        self.assertEqual("root",self.tdvp.state.orthogonality_center_id)
        self.assertEqual(ref_tdvp.state,self.tdvp.state)
        for identifiers in ref_tdvp.partial_tree_cache:
            correct_cache = ref_tdvp.partial_tree_cache.get_entry(identifiers[0],
                                                                          identifiers[1])
            found_cache = self.tdvp.partial_tree_cache.get_entry(identifiers[0],
                                                                         identifiers[1])
            self.assertTrue(np.allclose(correct_cache,found_cache))

        ref_tdvp.state.move_orthogonalization_center("c2",
                                                     mode=ptn.SplitMode.KEEP)
        update_root_to_c2_cache = contract_any("root", "c2",
                                               ref_tdvp.state, ref_tdvp.hamiltonian,
                                               ref_tdvp.partial_tree_cache)
        ref_tdvp.partial_tree_cache.add_entry("root","c2",
                                              update_root_to_c2_cache)

    def test_move_orth_and_update_cache_for_path_c1_to_c1(self):
        path = ["c1"]
        # In this case nothing should happen
        ref_tdvp = deepcopy(self.tdvp)
        self.tdvp._move_orth_and_update_cache_for_path(path)

        self.assertEqual("c1",self.tdvp.state.orthogonality_center_id)
        self.assertEqual(ref_tdvp.state,self.tdvp.state)
        for identifiers in ref_tdvp.partial_tree_cache:
            correct_cache = ref_tdvp.partial_tree_cache.get_entry(identifiers[0],
                                                                          identifiers[1])
            found_cache = self.tdvp.partial_tree_cache.get_entry(identifiers[0],
                                                                         identifiers[1])
            self.assertTrue(np.allclose(correct_cache,found_cache))

    def test_move_orth_and_update_cache_for_path_empty_path(self):
        path = []
        # In this case nothing should happen
        ref_tdvp = deepcopy(self.tdvp)
        self.tdvp._move_orth_and_update_cache_for_path(path)

        self.assertEqual("c1",self.tdvp.state.orthogonality_center_id)
        self.assertEqual(ref_tdvp.state,self.tdvp.state)
        for identifiers in ref_tdvp.partial_tree_cache:
            correct_cache = ref_tdvp.partial_tree_cache.get_entry(identifiers[0],
                                                                          identifiers[1])
            found_cache = self.tdvp.partial_tree_cache.get_entry(identifiers[0],
                                                                         identifiers[1])
            self.assertTrue(np.allclose(correct_cache,found_cache))

    def test_move_orth_and_update_cache_for_path_non_orth_center(self):
        self.assertRaises(AssertionError,
                          self.tdvp._move_orth_and_update_cache_for_path,
                          ["root","c2"])

    def test_contract_all_except_node_c1(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["c1"]
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("root", "c1")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,0,2,1])
        found_tensor = self.tdvp._contract_all_except_node("c1")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_c2(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["c2"]
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("root", "c2")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,0,2,1])

        found_tensor = self.tdvp._contract_all_except_node("c2")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_root(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["root"]
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("c1", "root")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("c2", "root")
        ref_tensor = np.tensordot(ref_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,5,0,2,4,1])

        found_tensor = self.tdvp._contract_all_except_node("root")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_get_effective_site_hamiltonian_c1(self):
        # Copmute Reference
        ref_tensor = np.tensordot(self.tdvp.partial_tree_cache.get_entry("root", "c1"),
                                  self.tdvp.hamiltonian.tensors["c1"],
                                  axes=(1,0))
        ref_tensor = np.transpose(ref_tensor, axes=(1,2,0,3))
        ref_tensor = np.reshape(ref_tensor, (15,15))

        found_tensor = self.tdvp._get_effective_site_hamiltonian("c1")
        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_site_hamiltonian_c2(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["c2"]
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("root", "c2")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,0,2,1])
        ref_tensor = np.reshape(ref_tensor, (24,24))

        found_tensor = self.tdvp._get_effective_site_hamiltonian("c2")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_get_effective_site_hamiltonian_root(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["root"]
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("c1", "root")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("c2", "root")
        ref_tensor = np.tensordot(ref_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,5,0,2,4,1])
        ref_tensor = np.reshape(ref_tensor, (60,60))

        found_tensor = self.tdvp._get_effective_site_hamiltonian("root")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_update_site_c1(self):
        node_id = "c1"
        ref_state = deepcopy(self.tdvp.state)
        eff_site_ham = self.tdvp._get_effective_site_hamiltonian(node_id)
        node_tensor = ref_state[node_id][1]
        node_state = np.reshape(node_tensor,15)
        exponent = expm(-1j * self.tdvp.time_step_size * eff_site_ham)
        ref_updated_state = exponent @ node_state
        ref_node_tensor = np.reshape(ref_updated_state,(5,3))

        self.tdvp._update_site(node_id)
        found_node_tensor = self.tdvp.state.tensors[node_id]
        self.assertTrue(np.allclose(ref_node_tensor, found_node_tensor))

    def test_update_site_root(self):
        node_id = "root"
        self.tdvp.state.move_orthogonalization_center(node_id,
                                                      mode=ptn.SplitMode.KEEP)
        self.tdvp.update_tree_cache("c1", node_id)
        ref_state = deepcopy(self.tdvp.state)
        eff_site_ham = self.tdvp._get_effective_site_hamiltonian(node_id)
        node_tensor = ref_state[node_id][1]
        node_state = np.reshape(node_tensor,60)
        exponent = expm(-1j * self.tdvp.time_step_size * eff_site_ham)
        ref_updated_state = exponent @ node_state
        ref_node_tensor = np.reshape(ref_updated_state,(6,5,2))

        self.tdvp._update_site(node_id)
        found_node_tensor = self.tdvp.state.tensors[node_id]
        self.assertTrue(np.allclose(ref_node_tensor, found_node_tensor))

    def test_update_site_c2(self):
        node_id = "c2"
        self.tdvp.state.move_orthogonalization_center(node_id,
                                                      mode=ptn.SplitMode.KEEP)
        self.tdvp.update_tree_cache("c1", "root")
        self.tdvp.update_tree_cache("root", node_id)
        ref_state = deepcopy(self.tdvp.state)
        eff_site_ham = self.tdvp._get_effective_site_hamiltonian(node_id)
        node_tensor = ref_state[node_id][1]
        node_state = np.reshape(node_tensor,24)
        exponent = expm(-1j * self.tdvp.time_step_size * eff_site_ham)
        ref_updated_state = exponent @ node_state
        ref_node_tensor = np.reshape(ref_updated_state,(6,4))

        self.tdvp._update_site(node_id)
        found_node_tensor = self.tdvp.state.tensors[node_id]
        self.assertTrue(np.allclose(ref_node_tensor, found_node_tensor))

class TestTDVPInitComplicated(unittest.TestCase):
    def setUp(self):
        self.ref_tree = ptn.random_big_ttns_two_root_children()
        self.hamiltonian = ptn.TTNO.from_hamiltonian(ptn.random_hamiltonian_compatible(),
                                                     self.ref_tree)
        self.tdvp = ptn.TDVPAlgorithm(self.ref_tree, self.hamiltonian, 0.1,1,
                                      ptn.TensorProduct({"site0": ptn.pauli_matrices()[0]}))

    def test_init_hamiltonian(self):
        self.assertEqual(self.hamiltonian, self.tdvp.hamiltonian)

    def test_init_state(self):
        self.assertEqual(self.ref_tree, self.tdvp.initial_state)

    def test_init_update_path(self):
        correct_path = ["site4","site5","site3","site2",
                        "site1","site0","site6","site7"]
        self.assertEqual(correct_path, self.tdvp.update_path)

    def test_init_orth_path(self):
        correct_path = [["site3","site5"],["site3"],["site1","site2"],
                        ["site1"],["site0"],["site6"],["site7"]]
        self.assertEqual(correct_path, self.tdvp.orthogonalization_path)

    def test_find_caching_path(self):
        caching_path, next_node_id_dict = self.tdvp._find_caching_path()
        correct_path = ["site7", "site6", "site0", "site2",
                        "site1", "site5", "site3", "site4"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site7": "site6", "site6": "site0",
                        "site0": "site1", "site2": "site1",
                        "site1": "site3", "site5": "site3", "site3": "site4"}
        self.assertEqual(correct_dict, next_node_id_dict)

    def test_init_tree_cache_7_to_6(self):
        node_id = "site7"
        next_node_id = "site6"
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(2,1))
        self.assertTrue((node_id,next_node_id) in self.tdvp.partial_tree_cache)

        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id,next_node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_init_tree_cache_2_to_1(self):
        node_id = "site2"
        next_node_id = "site1"
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(2,1))
        self.assertTrue((node_id,next_node_id) in self.tdvp.partial_tree_cache)

        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id,next_node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_init_tree_cache_5_to_3(self):
        node_id = "site2"
        next_node_id = "site1"
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(2,1))
        self.assertTrue((node_id,next_node_id) in self.tdvp.partial_tree_cache)

        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id,next_node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_init_tree_cache_6_to_0(self):
        node_id = "site6"
        next_node_id = "site0"
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(2,3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(4,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry("site7","site6"),
                                  axes=((1,3,5),(0,1,2)))
        self.assertTrue((node_id,next_node_id) in self.tdvp.partial_tree_cache)

        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id,next_node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_init_tree_cache_0_to_1(self):
        node_id = "site0"
        next_node_id = "site1"
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(2,3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(4,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry("site6","site0"),
                                  axes=((1,3,5),(0,1,2)))
        self.assertTrue((node_id,next_node_id) in self.tdvp.partial_tree_cache)

        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id,next_node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_init_tree_cache_1_to_3(self):
        node_id = "site1"
        next_node_id = "site3"
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.hamiltonian.tensors[node_id],
                                  axes=(3,4))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(6,3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry("site0",node_id),
                                  axes=((0,3,6),(0,1,2)))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry("site2",node_id),
                                  axes=((1,2,5),(0,1,2)))

        self.assertTrue((node_id,next_node_id) in self.tdvp.partial_tree_cache)
        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id,next_node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_init_tree_cache_3_to_4(self):
        node_id = "site3"
        next_node_id = "site4"
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.hamiltonian.tensors[node_id],
                                  axes=(3,4))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(6,3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry("site5",node_id),
                                  axes=((2,5,8),(0,1,2)))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry("site1",node_id),
                                  axes=((0,2,4),(0,1,2)))

        self.assertTrue((node_id,next_node_id) in self.tdvp.partial_tree_cache)
        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id,next_node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

class TestContractionMethodsComplicated(unittest.TestCase):
    """
    Tests the contraction methods of the TDVP algorithm for a complicated
     tree tensor network and not only for an MPS.
    """

    def setUp(self) -> None:
        ref_tree = ptn.random_big_ttns_two_root_children()
        hamiltonian = ptn.TTNO.from_hamiltonian(ptn.random_hamiltonian_compatible(),
                                                     ref_tree)
        self.tdvp = ptn.TDVPAlgorithm(ref_tree, hamiltonian, 0.1,1,
                                      ptn.TensorProduct({"site0": ptn.pauli_matrices()[0]}))
        # To correctly compute the contractions we need all potential cached tensors
        non_init_pairs = [("site4","site3"),("site3","site5"),("site3","site1"),
                          ("site1","site2"),("site1","site0"),("site0","site6"),
                          ("site6","site7")]
        for pair in non_init_pairs:
            self.tdvp.update_tree_cache(pair[0],pair[1])

    def test_find_tensor_leg_permutation_node_4(self):
        node_id = "site4"
        ref_perm = (3,0,2,1)
        found_perm = self.tdvp._find_tensor_leg_permutation(node_id)
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_5(self):
        node_id = "site5"
        ref_perm = (3,0,2,1)
        found_perm = self.tdvp._find_tensor_leg_permutation(node_id)
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_2(self):
        node_id = "site2"
        ref_perm = (3,0,2,1)
        found_perm = self.tdvp._find_tensor_leg_permutation(node_id)
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_7(self):
        node_id = "site7"
        ref_perm = (3,0,2,1)
        found_perm = self.tdvp._find_tensor_leg_permutation(node_id)
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_0(self):
        node_id = "site0"
        ref_perm = (3,5,0,2,4,1)
        found_perm = self.tdvp._find_tensor_leg_permutation(node_id)
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_6(self):
        node_id = "site6"
        ref_perm = (3,5,0,2,4,1)
        found_perm = self.tdvp._find_tensor_leg_permutation(node_id)
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_1(self):
        node_id = "site1"
        ref_perm = (3,7,5,0,2,6,4,1)
        found_perm = self.tdvp._find_tensor_leg_permutation(node_id)
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_3(self):
        node_id = "site3"
        ref_perm = (3,5,7,0,2,4,6,1)
        found_perm = self.tdvp._find_tensor_leg_permutation(node_id)
        self.assertEqual(ref_perm,found_perm)

    def test_contract_all_except_node_4(self):
        node_id = "site4"
        ref_tensor = np.tensordot(self.tdvp.partial_tree_cache.get_entry("site3",node_id),
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1,0))
        ref_tensor = np.transpose(ref_tensor, [1,2,0,3])

        found_tensor = self.tdvp._contract_all_except_node(node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_5(self):
        node_id = "site5"
        ref_tensor = np.tensordot(self.tdvp.partial_tree_cache.get_entry("site3",node_id),
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1,0))
        ref_tensor = np.transpose(ref_tensor, [1,2,0,3])

        found_tensor = self.tdvp._contract_all_except_node(node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_2(self):
        node_id = "site2"
        ref_tensor = np.tensordot(self.tdvp.partial_tree_cache.get_entry("site1",node_id),
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1,0))
        ref_tensor = np.transpose(ref_tensor, [1,2,0,3])

        found_tensor = self.tdvp._contract_all_except_node(node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_7(self):
        node_id = "site7"
        ref_tensor = np.tensordot(self.tdvp.partial_tree_cache.get_entry("site6",node_id),
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1,0))
        ref_tensor = np.transpose(ref_tensor, [1,2,0,3])

        found_tensor = self.tdvp._contract_all_except_node(node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_0(self):
        node_id = "site0"
        ham_tensor = self.tdvp.hamiltonian.tensors[node_id]
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site1",node_id)
        ref_tensor = np.tensordot(ham_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site6",node_id)
        ref_tensor = np.tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=(3,5,0,2,4,1))

        found_tensor = self.tdvp._contract_all_except_node(node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_6(self):
        node_id = "site6"
        ham_tensor = self.tdvp.hamiltonian.tensors[node_id]
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site0",node_id)
        ref_tensor = np.tensordot(ham_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site7",node_id)
        ref_tensor = np.tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=(3,5,0,2,4,1))

        found_tensor = self.tdvp._contract_all_except_node(node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_1(self):
        node_id = "site1"
        ham_tensor = self.tdvp.hamiltonian.tensors[node_id]
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site0",node_id)
        ref_tensor = np.tensordot(ham_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site2",node_id)
        ref_tensor = np.tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site3",node_id)
        ref_tensor = np.tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=(3,7,5,0,2,6,4,1))

        found_tensor = self.tdvp._contract_all_except_node(node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_3(self):
        node_id = "site3"
        ham_tensor = self.tdvp.hamiltonian.tensors[node_id]
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site1",node_id)
        ref_tensor = np.tensordot(ham_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site4",node_id)
        ref_tensor = np.tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_entry("site5",node_id)
        ref_tensor = np.tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=(3,5,7,0,2,4,6,1))

        found_tensor = self.tdvp._contract_all_except_node(node_id)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_get_effective_hamiltonian_4(self):
        node_id = "site4"
        ref_matrix = self.tdvp._contract_all_except_node(node_id)
        ref_matrix = np.reshape(ref_matrix,(4,4))

        found_matrix = self.tdvp._get_effective_site_hamiltonian(node_id)
        self.assertTrue(np.allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_5(self):
        node_id = "site5"
        ref_matrix = self.tdvp._contract_all_except_node(node_id)
        ref_matrix = np.reshape(ref_matrix,(4,4))

        found_matrix = self.tdvp._get_effective_site_hamiltonian(node_id)
        self.assertTrue(np.allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_2(self):
        node_id = "site2"
        ref_matrix = self.tdvp._contract_all_except_node(node_id)
        ref_matrix = np.reshape(ref_matrix,(4,4))

        found_matrix = self.tdvp._get_effective_site_hamiltonian(node_id)
        self.assertTrue(np.allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_7(self):
        node_id = "site7"
        ref_matrix = self.tdvp._contract_all_except_node(node_id)
        ref_matrix = np.reshape(ref_matrix,(4,4))

        found_matrix = self.tdvp._get_effective_site_hamiltonian(node_id)
        self.assertTrue(np.allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_0(self):
        node_id = "site0"
        ref_matrix = self.tdvp._contract_all_except_node(node_id)
        ref_matrix = np.reshape(ref_matrix,(8,8))

        found_matrix = self.tdvp._get_effective_site_hamiltonian(node_id)
        self.assertTrue(np.allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_6(self):
        node_id = "site6"
        ref_matrix = self.tdvp._contract_all_except_node(node_id)
        ref_matrix = np.reshape(ref_matrix,(8,8))

        found_matrix = self.tdvp._get_effective_site_hamiltonian(node_id)
        self.assertTrue(np.allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_1(self):
        node_id = "site1"
        ref_matrix = self.tdvp._contract_all_except_node(node_id)
        ref_matrix = np.reshape(ref_matrix,(16,16))

        found_matrix = self.tdvp._get_effective_site_hamiltonian(node_id)
        self.assertTrue(np.allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_3(self):
        node_id = "site3"
        ref_matrix = self.tdvp._contract_all_except_node(node_id)
        ref_matrix = np.reshape(ref_matrix,(16,16))

        found_matrix = self.tdvp._get_effective_site_hamiltonian(node_id)
        self.assertTrue(np.allclose(ref_matrix,found_matrix))

if __name__ == "__main__":
    unittest.main()
