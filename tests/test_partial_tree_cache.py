import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn

class TestInit(unittest.TestCase):

    def setUp(self) -> None:
        self.state = ptn.random_small_ttns()
        self.conversion_dict = {"root_op1": ptn.random_hermitian_matrix(),
                                "root_op2": ptn.random_hermitian_matrix(),
                                "I2": np.eye(2),
                                "c1_op": ptn.random_hermitian_matrix(size=3),
                                "I3": np.eye(3),
                                "c2_op": ptn.random_hermitian_matrix(size=4),
                                "I4": np.eye(4)}
        self.state = ptn.random_small_ttns()
        tensor_prod = [ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"})
                       ]
        ham = ptn.Hamiltonian(tensor_prod, self.conversion_dict)
        # All bond dim are 2
        self.hamiltonian = ptn.TTNO.from_hamiltonian(ham, self.state)

    def test_init_for_leaf_c1(self):
        node_id = "c1"
        # Computing the reference tensor
        state = deepcopy(self.state)
        hamiltonian = deepcopy(self.hamiltonian)
        ket_tensor = state.tensors[node_id]
        ref_tensor = np.tensordot(ket_tensor,
                                  hamiltonian.tensors[node_id],
                                   axes=([1],[2]))
        ref_tensor = np.tensordot(ref_tensor,
                                  ket_tensor.conj(),
                                  axes=([2],[1]))

        found_cache = ptn.PartialTreeCache.for_leaf(node_id,
                                                    self.state,
                                                    self.hamiltonian)

        self.assertEqual(state.nodes[node_id], found_cache.node)
        self.assertEqual("root", found_cache.pointing_to_node)
        self.assertTrue(np.allclose(ref_tensor, found_cache.tensor))

    def test_init_for_leaf_c2(self):
        node_id = "c2"
        # Computing the reference tensor
        state = deepcopy(self.state)
        hamiltonian = deepcopy(self.hamiltonian)
        ket_tensor = state.tensors[node_id]
        ref_tensor = np.tensordot(ket_tensor,
                                  hamiltonian.tensors[node_id],
                                   axes=([1],[2]))
        ref_tensor = np.tensordot(ref_tensor,
                                  ket_tensor.conj(),
                                  axes=([2],[1]))

        found_cache = ptn.PartialTreeCache.for_leaf(node_id,
                                                    self.state,
                                                    self.hamiltonian)

        self.assertEqual(state.nodes[node_id], found_cache.node)
        self.assertEqual("root", found_cache.pointing_to_node)
        self.assertTrue(np.allclose(ref_tensor, found_cache.tensor))

    def test_contract_neighbour_cache_to_ket(self):
        partial_tree_cache = ptn.PartialTreeChachDict(
                             {("c1", "root"):
                              ptn.PartialTreeCache.for_leaf("c1", self.state, self.hamiltonian),
                              ("c2", "root"):
                              ptn.PartialTreeCache.for_leaf("c2", self.state, self.hamiltonian)})
        node_id = "root"
        ket_tensor = deepcopy(self.state.tensors[node_id])
        ref_tensor = np.tensordot(ket_tensor,
                                  partial_tree_cache.get_entry("c2", node_id).tensor,
                                  axes=(1,0))

        cache = ptn.PartialTreeCache(self.state.nodes[node_id],
                                     "c1", self.state.tensors[node_id])
        cache._contract_neighbour_cache_to_ket("c2", 0,
                                               partial_tree_cache)

        self.assertTrue(np.allclose(ref_tensor, cache.tensor))

    def test_contract_hamiltonian_tensor(self):
        partial_tree_cache = ptn.PartialTreeChachDict(
                             {("c1", "root"):
                              ptn.PartialTreeCache.for_leaf("c1", self.state, self.hamiltonian),
                              ("c2", "root"):
                              ptn.PartialTreeCache.for_leaf("c2", self.state, self.hamiltonian)})
        node_id = "root"
        ket_tensor = deepcopy(self.state.tensors[node_id])
        ham_tensor = deepcopy(self.hamiltonian.tensors[node_id])
        ref_tensor = np.tensordot(ket_tensor,
                                  partial_tree_cache.get_entry("c2", node_id).tensor,
                                  axes=(1,0))
        ref_tensor = np.tensordot(ref_tensor,
                                  ham_tensor,
                                  axes=([1,2],[2,1]))

        cache = ptn.PartialTreeCache(self.state.nodes[node_id],
                                     "c1", self.state.tensors[node_id])
        cache._contract_neighbour_cache_to_ket("c2", 0,
                                               partial_tree_cache)
        cache._contract_hamiltonian_tensor(0, self.hamiltonian.tensors[node_id])

        self.assertTrue(np.allclose(ref_tensor, cache.tensor))

    def test_init_with_existing_cache_for_root_c1(self):
        partial_tree_cache = ptn.PartialTreeChachDict(
                             {("c1", "root"):
                              ptn.PartialTreeCache.for_leaf("c1", self.state, self.hamiltonian),
                              ("c2", "root"):
                              ptn.PartialTreeCache.for_leaf("c2", self.state, self.hamiltonian)})
        # Compute reference tensor
        node_id = "root"
        ket_tensor = deepcopy(self.state.tensors[node_id])
        ham_tensor = deepcopy(self.hamiltonian.tensors[node_id])
        ref_tensor = np.tensordot(ket_tensor,
                                  partial_tree_cache.get_entry("c2", node_id).tensor,
                                  axes=(1,0))
        ref_tensor = np.tensordot(ref_tensor,
                                  ham_tensor,
                                  axes=([1,2],[2,1]))
        ref_tensor = np.tensordot(ref_tensor,
                                  ket_tensor.conj(),
                                  axes=([1,3],[1,2]))

        found_cache = ptn.PartialTreeCache.with_existing_cache("root",
                                                               "c1",
                                                               partial_tree_cache,
                                                               self.state,
                                                               self.hamiltonian)

        self.assertTrue(np.allclose(ref_tensor, found_cache.tensor))

    def test_init_with_existing_cache_for_root_c2(self):
        partial_tree_cache = ptn.PartialTreeChachDict(
                             {("c1", "root"):
                              ptn.PartialTreeCache.for_leaf("c1", self.state, self.hamiltonian),
                              ("c2", "root"):
                              ptn.PartialTreeCache.for_leaf("c2", self.state, self.hamiltonian)})
        # Compute reference tensor
        node_id = "root"
        ket_tensor = deepcopy(self.state.tensors[node_id])
        ham_tensor = deepcopy(self.hamiltonian.tensors[node_id])
        ref_tensor = np.tensordot(ket_tensor,
                                  partial_tree_cache.get_entry("c1", node_id).tensor,
                                  axes=(0,0))
        ref_tensor = np.tensordot(ref_tensor,
                                  ham_tensor,
                                  axes=([1,2],[2,0]))
        ref_tensor = np.tensordot(ref_tensor,
                                  ket_tensor.conj(),
                                  axes=([1,3],[0,2]))

        found_cache = ptn.PartialTreeCache.with_existing_cache("root",
                                                               "c2",
                                                               partial_tree_cache,
                                                               self.state,
                                                               self.hamiltonian)

        self.assertTrue(np.allclose(ref_tensor, found_cache.tensor))

if __name__ == "__main__":
    unittest.main()