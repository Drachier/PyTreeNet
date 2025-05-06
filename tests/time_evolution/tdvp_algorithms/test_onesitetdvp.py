
import unittest
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

import pytreenet as ptn
from pytreenet.contractions.state_operator_contraction import (contract_any)
from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.random import (random_hermitian_matrix,
                              random_small_ttns,
                              random_big_ttns_two_root_children,
                              random_hamiltonian_compatible,
                              crandn)
from pytreenet.util import (compute_transfer_tensor,
                            SplitMode,
                            NoConnectionException)

class TestContractionMethods(unittest.TestCase):

    def setUp(self):
        self.conversion_dict = {"root_op1": random_hermitian_matrix(),
                                "root_op2": random_hermitian_matrix(),
                                "I2": np.eye(2),
                                "c1_op": random_hermitian_matrix(size=3),
                                "I3": np.eye(3),
                                "c2_op": random_hermitian_matrix(size=4),
                                "I4": np.eye(4)}
        self.ref_tree = random_small_ttns()
        tensor_prod = [ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct(
                           {"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct(
                           {"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
                       ptn.TensorProduct(
                           {"c1": "c1_op", "root": "I2", "c2": "c2_op"})
                       ]
        ham = ptn.Hamiltonian(tensor_prod, self.conversion_dict)
        operator = ptn.TensorProduct({"root": crandn((2, 2))})
        self.hamiltonian = ptn.TTNO.from_hamiltonian(ham, self.ref_tree)
        self.tdvp = ptn.OneSiteTDVP(self.ref_tree, self.hamiltonian,
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

    def test_get_effective_link_hamiltonian_c1_to_root(self):
        root_id = "root"
        node_id = "c1"
        self.tdvp._split_updated_site(node_id, root_id)
        cache_c1 = self.tdvp.partial_tree_cache.get_entry(root_id, node_id)
        cache_root = self.tdvp.partial_tree_cache.get_entry(node_id, root_id)
        ref_tensor = np.tensordot(cache_c1, cache_root,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (25, 25))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            node_id, root_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_link_hamiltonian_c2_to_root(self):
        root_id = "root"
        node_id = "c2"
        self.tdvp._split_updated_site(node_id, root_id)
        cache_c1 = self.tdvp.partial_tree_cache.get_entry(root_id, node_id)
        cache_root = self.tdvp.partial_tree_cache.get_entry(node_id, root_id)
        ref_tensor = np.tensordot(cache_c1, cache_root,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (36, 36))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            node_id, root_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_link_hamiltonian_root_to_c1(self):
        root_id = "root"
        node_id = "c1"
        self.tdvp._split_updated_site(root_id, node_id)
        cache_c1 = self.tdvp.partial_tree_cache.get_entry(node_id, root_id)
        cache_root = self.tdvp.partial_tree_cache.get_entry(root_id, node_id)
        ref_tensor = np.tensordot(cache_root, cache_c1,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (25, 25))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            root_id, node_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_link_hamiltonian_root_to_c2(self):
        root_id = "root"
        node_id = "c2"
        self.tdvp._split_updated_site(root_id, node_id)
        cache_c1 = self.tdvp.partial_tree_cache.get_entry(node_id, root_id)
        cache_root = self.tdvp.partial_tree_cache.get_entry(root_id, node_id)
        ref_tensor = np.tensordot(cache_root, cache_c1,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (36, 36))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            root_id, node_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_time_evolve_link_tensor_c1_to_root(self):
        node_id = "c1"
        next_node_id = "root"
        ref_tdvp = deepcopy(self.tdvp)
        ref_tdvp._split_updated_site(node_id, next_node_id)
        link_id = ref_tdvp.create_link_id(node_id, next_node_id)
        ref_tensor = ref_tdvp.state[link_id][1]
        ref_tensor = np.reshape(ref_tensor, 25)
        eff_link_ham = ref_tdvp._get_effective_link_hamiltonian(
            node_id, next_node_id)
        exponent = expm(1j * ref_tdvp.time_step_size * eff_link_ham)
        updated_ref_tensor = exponent @ ref_tensor
        updated_ref_tensor = np.reshape(updated_ref_tensor, (5, 5))
        ref_tdvp.state.tensors[link_id] = updated_ref_tensor

        self.tdvp._split_updated_site(node_id, next_node_id)
        self.tdvp._time_evolve_link_tensor(node_id, next_node_id)
        self.assertTrue(np.allclose(ref_tdvp.state.tensors[node_id],
                                    self.tdvp.state.tensors[node_id]))
        self.assertTrue(np.allclose(ref_tdvp.state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_update_cache_after_split_c1_to_root(self):
        node_id = "c1"
        q_legs = ptn.LegSpecification(None, [], [1])
        r_legs = ptn.LegSpecification("root", [], [])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id, "root"))

        # Compute Reference
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1, 2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(2, 1))

        self.tdvp._update_cache_after_split(node_id, "root")
        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id, "root")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_update_cache_after_split_c2_to_root(self):
        node_id = "c2"
        q_legs = ptn.LegSpecification(None, [], [1])
        r_legs = ptn.LegSpecification("root", [], [])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id, "root"))

        # Compute Reference
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1, 2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(2, 1))

        self.tdvp._update_cache_after_split(node_id, "root")
        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id, "root")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_update_cache_after_split_root_to_c1(self):
        node_id = "root"
        q_legs = ptn.LegSpecification(None, ["c2"], [2], is_root=True)
        r_legs = ptn.LegSpecification(None, ["c1"], [])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id, "c1"))

        # Compute Reference
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(2, 3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(4, 2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry(
                                      "c2", node_id),
                                  axes=([1, 3, 5], [0, 1, 2]))

        self.tdvp._update_cache_after_split(node_id, "c1")
        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id, "c1")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_update_cache_after_split_root_to_c2(self):
        node_id = "root"
        q_legs = ptn.LegSpecification(None, ["c1"], [2], is_root=True)
        r_legs = ptn.LegSpecification(None, ["c2"], [])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id, "c2"))
        # Compute Reference
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(2, 3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(4, 2))
        cached_node = self.tdvp.partial_tree_cache.get_entry("c1", node_id)
        # Note the link tensor is the first child of root not c1!
        ref_tensor = np.tensordot(ref_tensor,
                                  cached_node,
                                  axes=([1, 2, 5], [0, 1, 2]))

        self.tdvp._update_cache_after_split(node_id, "c2")
        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id, "c2")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_split_updated_site_c1(self):
        node_id = "c1"
        q_legs = ptn.LegSpecification(None, [], [1])
        r_legs = ptn.LegSpecification("root", [], [])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(node_id, q_legs, r_legs,
                                q_identifier=node_id,
                                r_identifier=self.tdvp.create_link_id(
                                    node_id, "root"),
                                mode=SplitMode.KEEP)
        ref_old_cache = deepcopy(
            self.tdvp.partial_tree_cache.get_entry(node_id, "root"))

        self.tdvp._split_updated_site(node_id, "root")
        self.assertTrue(np.allclose(ref_state.tensors[node_id],
                                    self.tdvp.state.tensors[node_id]))
        self.assertFalse(np.allclose(ref_old_cache,
                                     self.tdvp.partial_tree_cache.get_entry(node_id, "root")))
        link_id = self.tdvp.create_link_id(node_id, "root")
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_split_updated_site_c2(self):
        node_id = "c2"
        q_legs = ptn.LegSpecification(None, [], [1])
        r_legs = ptn.LegSpecification("root", [], [])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(node_id, q_legs, r_legs,
                                q_identifier=node_id,
                                r_identifier=self.tdvp.create_link_id(
                                    node_id, "root"),
                                mode=SplitMode.KEEP)

        self.tdvp._split_updated_site(node_id, "root")
        self.assertTrue(np.allclose(ref_state.tensors[node_id],
                                    self.tdvp.state.tensors[node_id]))
        link_id = self.tdvp.create_link_id(node_id, "root")
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_split_updated_site_root_to_c1(self):
        node_id = "c1"
        root_id = "root"
        q_legs = ptn.LegSpecification(None, ["c2"], [2], is_root=True)
        r_legs = ptn.LegSpecification(None, [node_id], [])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(root_id, q_legs, r_legs,
                                q_identifier=root_id,
                                r_identifier=self.tdvp.create_link_id(root_id, node_id))

        self.tdvp._split_updated_site(root_id, node_id)
        self.assertTrue(np.allclose(ref_state.tensors[root_id],
                                    self.tdvp.state.tensors[root_id]))
        link_id = self.tdvp.create_link_id(root_id, node_id)
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_split_updated_site_root_to_c2(self):
        node_id = "c2"
        root_id = "root"
        q_legs = ptn.LegSpecification(None, ["c1"], [2], is_root=True)
        r_legs = ptn.LegSpecification(None, [node_id], [])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(root_id, q_legs, r_legs,
                                q_identifier=root_id,
                                r_identifier=self.tdvp.create_link_id(root_id, node_id))

        self.tdvp._split_updated_site(root_id, node_id)
        self.assertTrue(np.allclose(ref_state.tensors[root_id],
                                    self.tdvp.state.tensors[root_id]))
        link_id = self.tdvp.create_link_id(root_id, node_id)
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_split_updated_site_exception(self):
        self.assertRaises(NoConnectionException,
                          self.tdvp._split_updated_site,
                          "c1", "c2")

    def test_update_link_wrong_orth_center(self):
        self.assertRaises(AssertionError, self.tdvp._update_link,
                          "root", "c2")

    def test_update_link_c1_to_root(self):
        node_id = "c1"
        next_node_id = "root"
        ref_tdvp = deepcopy(self.tdvp)
        ref_tdvp._split_updated_site(node_id, next_node_id)
        link_id = ref_tdvp.create_link_id(node_id, next_node_id)
        ref_tensor = ref_tdvp.state[link_id][1]
        ref_tensor = np.reshape(ref_tensor, 25)
        eff_link_ham = ref_tdvp._get_effective_link_hamiltonian(
            node_id, next_node_id)
        exponent = expm(1j * ref_tdvp.time_step_size * eff_link_ham)
        updated_ref_tensor = exponent @ ref_tensor
        updated_ref_tensor = np.reshape(updated_ref_tensor, (5, 5))
        ref_tdvp.state.tensors[link_id] = updated_ref_tensor
        ref_tdvp.state.contract_nodes(link_id, next_node_id,
                                      new_identifier=next_node_id)
        ref_tdvp.state.orthogonality_center_id = next_node_id

        self.tdvp._update_link(node_id, next_node_id)
        transfer_tensor = compute_transfer_tensor(ref_tdvp.state.tensors[node_id],
                                                      (0, ))
        self.assertTrue(np.allclose(np.eye(3), transfer_tensor))
        self.assertEqual(ref_tdvp.state, self.tdvp.state)

    def test_update_link_root_to_c1(self):
        node_id = "root"
        next_node_id = "c1"
        self.tdvp._move_orth_and_update_cache_for_path(["c1", "root"])
        ref_tdvp = deepcopy(self.tdvp)
        ref_tdvp._split_updated_site(node_id, next_node_id)
        link_id = ref_tdvp.create_link_id(node_id, next_node_id)
        ref_link_tensor = ref_tdvp.state.tensors[link_id]
        orig_shape = ref_link_tensor.shape
        ref_link_tensor = np.reshape(ref_link_tensor,
                                     np.prod(orig_shape))
        eff_link_ham = ref_tdvp._get_effective_link_hamiltonian(node_id,
                                                                next_node_id)
        exponent = expm(1j*ref_tdvp.time_step_size*eff_link_ham)
        updated_ref_tensor = exponent @ ref_link_tensor
        updated_ref_tensor = np.reshape(updated_ref_tensor, orig_shape)
        ref_tdvp.state.tensors[link_id] = updated_ref_tensor
        ref_tdvp.state.contract_nodes(link_id, next_node_id,
                                      new_identifier=next_node_id)
        ref_tdvp.state.orthogonality_center_id = next_node_id

        self.tdvp._update_link(node_id, next_node_id)
        self.assertEqual(ref_tdvp.state, self.tdvp.state)

    def test_update_link_root_to_c2(self):
        node_id = "root"
        next_node_id = "c2"
        self.tdvp._move_orth_and_update_cache_for_path(["c1", "root"])
        ref_tdvp = deepcopy(self.tdvp)
        ref_tdvp._split_updated_site(node_id, next_node_id)
        link_id = ref_tdvp.create_link_id(node_id, next_node_id)
        ref_link_tensor = ref_tdvp.state.tensors[link_id]
        orig_shape = ref_link_tensor.shape
        ref_link_tensor = np.reshape(ref_link_tensor,
                                     np.prod(orig_shape))
        eff_link_ham = ref_tdvp._get_effective_link_hamiltonian(node_id,
                                                                next_node_id)
        exponent = expm(1j*ref_tdvp.time_step_size*eff_link_ham)
        updated_ref_tensor = exponent @ ref_link_tensor
        updated_ref_tensor = np.reshape(updated_ref_tensor, orig_shape)
        ref_tdvp.state.tensors[link_id] = updated_ref_tensor
        ref_tdvp.state.contract_nodes(link_id, next_node_id,
                                      new_identifier=next_node_id)
        ref_tdvp.state.orthogonality_center_id = next_node_id

        self.tdvp._update_link(node_id, next_node_id)
        self.assertEqual(ref_tdvp.state, self.tdvp.state)

    def test_update_link_c2_to_root(self):
        node_id = "c2"
        next_node_id = "root"
        self.tdvp._move_orth_and_update_cache_for_path(["c1", "root", "c2"])
        ref_tdvp = deepcopy(self.tdvp)
        ref_tdvp._split_updated_site(node_id, next_node_id)
        link_id = ref_tdvp.create_link_id(node_id, next_node_id)
        ref_link_tensor = ref_tdvp.state.tensors[link_id]
        orig_shape = ref_link_tensor.shape
        ref_link_tensor = np.reshape(ref_link_tensor,
                                     np.prod(orig_shape))
        eff_link_ham = ref_tdvp._get_effective_link_hamiltonian(node_id,
                                                                next_node_id)
        exponent = expm(1j*ref_tdvp.time_step_size*eff_link_ham)
        updated_ref_tensor = exponent @ ref_link_tensor
        updated_ref_tensor = np.reshape(updated_ref_tensor, orig_shape)
        ref_tdvp.state.tensors[link_id] = updated_ref_tensor
        ref_tdvp.state.contract_nodes(link_id, next_node_id,
                                      new_identifier=next_node_id)
        ref_tdvp.state.orthogonality_center_id = next_node_id

        self.tdvp._update_link(node_id, next_node_id)
        self.assertEqual(ref_tdvp.state, self.tdvp.state)


class TestContractionMethodsComplicated(unittest.TestCase):
    """
    Tests the contraction methods of the TDVP algorithm for a complicated
     tree tensor network and not only for an MPS.
    """

    def setUp(self) -> None:
        ref_tree = random_big_ttns_two_root_children()
        hamiltonian = ptn.TTNO.from_hamiltonian(random_hamiltonian_compatible(),
                                                ref_tree)
        self.tdvp = ptn.OneSiteTDVP(ref_tree, hamiltonian, 0.1, 1,
                                    ptn.TensorProduct({"site0": pauli_matrices()[0]}))
        # To correctly compute the contractions we need all potential cached tensors
        non_init_pairs = [("site4", "site3"), ("site3", "site5"), ("site3", "site1"),
                          ("site1", "site2"), ("site1",
                                               "site0"), ("site0", "site6"),
                          ("site6", "site7")]
        for pair in non_init_pairs:
            self.tdvp.update_tree_cache(pair[0], pair[1])

    def test_get_effective_link_hamiltonian_1_to_2(self):
        node_id = "site1"
        next_node_id = "site2"
        self.tdvp._split_updated_site(node_id, next_node_id)
        cache_1 = self.tdvp.partial_tree_cache.get_entry(next_node_id, node_id)
        cache_2 = self.tdvp.partial_tree_cache.get_entry(node_id, next_node_id)
        ref_tensor = np.tensordot(cache_2, cache_1,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (4, 4))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            node_id, next_node_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_link_hamiltonian_2_to_1(self):
        node_id = "site2"
        next_node_id = "site1"
        self.tdvp._split_updated_site(node_id, next_node_id)
        cache_1 = self.tdvp.partial_tree_cache.get_entry(next_node_id, node_id)
        cache_2 = self.tdvp.partial_tree_cache.get_entry(node_id, next_node_id)
        ref_tensor = np.tensordot(cache_1, cache_2,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (4, 4))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            node_id, next_node_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_link_hamiltonian_1_to_3(self):
        node_id = "site1"
        next_node_id = "site3"
        self.tdvp._split_updated_site(node_id, next_node_id)
        cache_1 = self.tdvp.partial_tree_cache.get_entry(next_node_id, node_id)
        cache_2 = self.tdvp.partial_tree_cache.get_entry(node_id, next_node_id)
        ref_tensor = np.tensordot(cache_2, cache_1,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (4, 4))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            node_id, next_node_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_link_hamiltonian_3_to_1(self):
        node_id = "site3"
        next_node_id = "site1"
        self.tdvp._split_updated_site(node_id, next_node_id)
        cache_1 = self.tdvp.partial_tree_cache.get_entry(next_node_id, node_id)
        cache_2 = self.tdvp.partial_tree_cache.get_entry(node_id, next_node_id)
        ref_tensor = np.tensordot(cache_1, cache_2,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (4, 4))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            node_id, next_node_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_link_hamiltonian_1_to_0(self):
        node_id = "site1"
        next_node_id = "site0"
        self.tdvp._split_updated_site(node_id, next_node_id)
        cache_1 = self.tdvp.partial_tree_cache.get_entry(next_node_id, node_id)
        cache_2 = self.tdvp.partial_tree_cache.get_entry(node_id, next_node_id)
        ref_tensor = np.tensordot(cache_1, cache_2,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (4, 4))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            node_id, next_node_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_link_hamiltonian_0_to_1(self):
        node_id = "site0"
        next_node_id = "site1"
        self.tdvp._split_updated_site(node_id, next_node_id)
        cache_1 = self.tdvp.partial_tree_cache.get_entry(next_node_id, node_id)
        cache_2 = self.tdvp.partial_tree_cache.get_entry(node_id, next_node_id)
        ref_tensor = np.tensordot(cache_2, cache_1,
                                  axes=(1, 1))
        ref_tensor = np.transpose(ref_tensor, axes=[1, 3, 0, 2])
        ref_tensor = np.reshape(ref_tensor, (4, 4))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(
            node_id, next_node_id)

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_update_cache_after_1_root_to_2(self):
        node_id = "site1"
        q_legs = ptn.LegSpecification("site0", ["site3"], [3])
        r_legs = ptn.LegSpecification(None, ["site2"], [])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id, "site2"))
        # Compute Reference
        # Note that the qr decomposition caused the children to flip again
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(3, 4))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(6, 3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry(
                                      "site0", node_id),
                                  axes=([0, 3, 6], [0, 1, 2]))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry(
                                      "site3", node_id),
                                  axes=([1, 3, 5], [0, 1, 2]))

        self.tdvp._update_cache_after_split(node_id, "site2")
        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id, "site2")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_update_cache_after_1_root_to_3(self):
        node_id = "site1"
        q_legs = ptn.LegSpecification("site0", ["site2"], [3])
        r_legs = ptn.LegSpecification(None, ["site3"], [])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id, "site3"))
        # Compute Reference
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(3, 4))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(6, 3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry(
                                      "site0", node_id),
                                  axes=([0, 3, 6], [0, 1, 2]))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_entry(
                                      "site2", node_id),
                                  axes=([1, 2, 5], [0, 1, 2]))

        self.tdvp._update_cache_after_split(node_id, "site3")
        found_tensor = self.tdvp.partial_tree_cache.get_entry(node_id, "site3")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_split_updated_site_1_to_2(self):
        node_id = "site1"
        next_node_id = "site2"
        q_legs = ptn.LegSpecification("site0", ["site3"], [3])
        r_legs = ptn.LegSpecification(None, ["site2"], [])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(node_id, q_legs, r_legs,
                                q_identifier=node_id,
                                r_identifier=self.tdvp.create_link_id(node_id, next_node_id))

        self.tdvp._split_updated_site(node_id, next_node_id)
        self.assertTrue(np.allclose(ref_state.tensors[node_id],
                                    self.tdvp.state.tensors[node_id]))
        link_id = self.tdvp.create_link_id(node_id, next_node_id)
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_split_updated_site_1_to_3(self):
        node_id = "site1"
        next_node_id = "site3"
        q_legs = ptn.LegSpecification("site0", ["site2"], [3])
        r_legs = ptn.LegSpecification(None, ["site3"], [])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(node_id, q_legs, r_legs,
                                q_identifier=node_id,
                                r_identifier=self.tdvp.create_link_id(node_id, next_node_id))

        self.tdvp._split_updated_site(node_id, next_node_id)
        self.assertTrue(np.allclose(ref_state.tensors[node_id],
                                    self.tdvp.state.tensors[node_id]))
        link_id = self.tdvp.create_link_id(node_id, next_node_id)
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))


if __name__ == "__main__":
    unittest.main()
