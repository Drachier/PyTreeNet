import unittest

from numpy import tensordot, transpose, allclose, reshape

from pytreenet.core.graph_node import GraphNode
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.contractions.sandwich_caching import SandwichCache

from pytreenet.contractions.effective_hamiltonians import (find_tensor_leg_permutation,
                                                           contract_all_except_node,
                                                           get_effective_single_site_hamiltonian)

class TestContractionMethodsSimple(unittest.TestCase):

    def setUp(self):
        self.ttns, self.hamiltonian = small_ttns_and_ttno()
        self.cache = SandwichCache(self.ttns, self.hamiltonian)
        # Cached tensors for use
        self.cache.update_tree_cache("c1","root")
        self.cache.update_tree_cache("c2","root")
        self.cache.update_tree_cache("root","c1")
        self.cache.update_tree_cache("root","c2")

    def test_find_tensor_leg_permutation_trivial(self):
        state_node = GraphNode()
        state_node.add_parent("root")
        state_node.add_children(["c1","c2"])
        hamiltonian_node = GraphNode()
        hamiltonian_node.add_parent("root")
        hamiltonian_node.add_children(["c1","c2"])
        permutation = find_tensor_leg_permutation(state_node,
                                                    hamiltonian_node)
        correct_perm = tuple([3,5,7,0,2,4,6,1])
        self.assertEqual(permutation, correct_perm)

    def test_find_tensor_leg_permutation_diff(self):
        state_node = GraphNode()
        state_node.add_parent("root")
        state_node.add_children(["c1","c2"])
        hamiltonian_node = GraphNode()
        hamiltonian_node.add_parent("root")
        hamiltonian_node.add_children(["c2","c1"])
        permutation = find_tensor_leg_permutation(state_node,
                                                    hamiltonian_node)
        correct_perm = tuple([3,7,5,0,2,6,4,1])
        self.assertEqual(permutation, correct_perm)

    def test_contract_all_except_node_c1(self):
        # Compute Reference
        ham_tensor = self.hamiltonian.tensors["c1"]
        cache_tensor = self.cache.get_entry("root", "c1")
        ref_tensor = tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = transpose(ref_tensor, axes=[3,0,2,1])
        found_tensor = contract_all_except_node("c1",
                                                self.ttns,
                                                self.hamiltonian,
                                                self.cache)

        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_c2(self):
        # Compute Reference
        ham_tensor = self.hamiltonian.tensors["c2"]
        cache_tensor = self.cache.get_entry("root", "c2")
        ref_tensor = tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = transpose(ref_tensor, axes=[3,0,2,1])

        found_tensor = contract_all_except_node("c2",
                                                self.ttns,
                                                self.hamiltonian,
                                                self.cache)

        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_root(self):
        # Compute Reference
        ham_tensor = self.hamiltonian.tensors["root"]
        cache_tensor = self.cache.get_entry("c1", "root")
        ref_tensor = tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.cache.get_entry("c2", "root")
        ref_tensor = tensordot(ref_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = transpose(ref_tensor, axes=[3,5,0,2,4,1])

        found_tensor = contract_all_except_node("root",
                                                self.ttns,
                                                self.hamiltonian,
                                                self.cache)


        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_get_effective_site_hamiltonian_c1(self):
        # Copmute Reference
        ref_tensor = tensordot(self.cache.get_entry("root", "c1"),
                                  self.hamiltonian.tensors["c1"],
                                  axes=(1,0))
        ref_tensor = transpose(ref_tensor, axes=(1,2,0,3))
        ref_tensor = reshape(ref_tensor, (15,15))

        found_tensor = get_effective_single_site_hamiltonian("c1",
                                                            self.ttns,
                                                            self.hamiltonian,
                                                            self.cache)

        self.assertTrue(allclose(ref_tensor, found_tensor))

    def test_get_effective_site_hamiltonian_c2(self):
        # Compute Reference
        ham_tensor = self.hamiltonian.tensors["c2"]
        cache_tensor = self.cache.get_entry("root", "c2")
        ref_tensor = tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = transpose(ref_tensor, axes=[3,0,2,1])
        ref_tensor = reshape(ref_tensor, (24,24))

        found_tensor = get_effective_single_site_hamiltonian("c2",
                                                            self.ttns,
                                                            self.hamiltonian,
                                                            self.cache)

        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_get_effective_site_hamiltonian_root(self):
        # Compute Reference
        ham_tensor = self.hamiltonian.tensors["root"]
        cache_tensor = self.cache.get_entry("c1", "root")
        ref_tensor = tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.cache.get_entry("c2", "root")
        ref_tensor = tensordot(ref_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = transpose(ref_tensor, axes=[3,5,0,2,4,1])
        ref_tensor = reshape(ref_tensor, (60,60))

        found_tensor = get_effective_single_site_hamiltonian("root",
                                                            self.ttns,
                                                            self.hamiltonian,
                                                            self.cache)

        self.assertTrue(allclose(ref_tensor,found_tensor))

class TestContractionMethodsBig(unittest.TestCase):
    def setUp(self) -> None:
        self.state, self.hamiltonian = big_ttns_and_ttno()
        self.state.canonical_form("site4") # To have some nodes with differeing child ordering
        self.cache = SandwichCache(self.state, self.hamiltonian)
        # To correctly compute the contractions we need all potential cached tensors
        non_init_pairs = [("site4","site3"),("site5","site3"),("site3","site1"),
                          ("site2","site1"),("site1","site0"),("site0","site6"),
                          ("site6","site7"),
                          ("site7","site6"),("site6","site0"),("site0","site1"),
                          ("site1","site2"),("site1","site3"),("site3","site4"),
                          ("site3","site5")]
        for pair in non_init_pairs:
            self.cache.update_tree_cache(pair[0],pair[1])

    def test_find_tensor_leg_permutation_node_4(self):
        node_id = "site4"
        ref_perm = (3,0,2,1)
        found_perm = find_tensor_leg_permutation(self.state.nodes[node_id],
                                                 self.hamiltonian.nodes[node_id])
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_5(self):
        node_id = "site5"
        ref_perm = (3,0,2,1)
        found_perm = find_tensor_leg_permutation(self.state.nodes[node_id],
                                                 self.hamiltonian.nodes[node_id])
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_2(self):
        node_id = "site2"
        ref_perm = (3,0,2,1)
        found_perm = find_tensor_leg_permutation(self.state.nodes[node_id],
                                                 self.hamiltonian.nodes[node_id])
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_7(self):
        node_id = "site7"
        ref_perm = (3,0,2,1)
        found_perm = find_tensor_leg_permutation(self.state.nodes[node_id],
                                                 self.hamiltonian.nodes[node_id])
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_0(self):
        node_id = "site0"
        ref_perm = (3,5,0,2,4,1)
        found_perm = find_tensor_leg_permutation(self.state.nodes[node_id],
                                                 self.hamiltonian.nodes[node_id])
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_6(self):
        node_id = "site6"
        ref_perm = (3,5,0,2,4,1)
        found_perm = find_tensor_leg_permutation(self.state.nodes[node_id],
                                                 self.hamiltonian.nodes[node_id])
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_1(self):
        node_id = "site1"
        ref_perm = (3,7,5,0,2,6,4,1)
        found_perm = find_tensor_leg_permutation(self.state.nodes[node_id],
                                                 self.hamiltonian.nodes[node_id])
        self.assertEqual(ref_perm,found_perm)

    def test_find_tensor_leg_permutation_node_3(self):
        node_id = "site3"
        ref_perm = (3,5,7,0,2,4,6,1)
        found_perm = find_tensor_leg_permutation(self.state.nodes[node_id],
                                                 self.hamiltonian.nodes[node_id])
        self.assertEqual(ref_perm,found_perm)

    def test_contract_all_except_node_4(self):
        node_id = "site4"
        ref_tensor = tensordot(self.cache.get_entry("site3",node_id),
                                  self.hamiltonian.tensors[node_id],
                                  axes=(1,0))
        ref_tensor = transpose(ref_tensor, [1,2,0,3])

        found_tensor = contract_all_except_node(node_id,
                                                     self.state,
                                                     self.hamiltonian,
                                                     self.cache)
        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_5(self):
        node_id = "site5"
        ref_tensor = tensordot(self.cache.get_entry("site3",node_id),
                                  self.hamiltonian.tensors[node_id],
                                  axes=(1,0))
        ref_tensor = transpose(ref_tensor, [1,2,0,3])

        found_tensor = contract_all_except_node(node_id,
                                                     self.state,
                                                     self.hamiltonian,
                                                     self.cache)
        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_2(self):
        node_id = "site2"
        ref_tensor = tensordot(self.cache.get_entry("site1",node_id),
                                  self.hamiltonian.tensors[node_id],
                                  axes=(1,0))
        ref_tensor = transpose(ref_tensor, [1,2,0,3])

        found_tensor = contract_all_except_node(node_id,
                                                     self.state,
                                                     self.hamiltonian,
                                                     self.cache)
        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_7(self):
        node_id = "site7"
        ref_tensor = tensordot(self.cache.get_entry("site6",node_id),
                                  self.hamiltonian.tensors[node_id],
                                  axes=(1,0))
        ref_tensor = transpose(ref_tensor, [1,2,0,3])

        found_tensor = contract_all_except_node(node_id,
                                                     self.state,
                                                     self.hamiltonian,
                                                     self.cache)
        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_0(self):
        node_id = "site0"
        ham_tensor = self.hamiltonian.tensors[node_id]
        cache_tensor = self.cache.get_entry("site1",node_id)
        ref_tensor = tensordot(ham_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.cache.get_entry("site6",node_id)
        ref_tensor = tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        ref_tensor = transpose(ref_tensor, axes=(3,5,0,2,4,1))

        found_tensor = contract_all_except_node(node_id,
                                                     self.state,
                                                     self.hamiltonian,
                                                     self.cache)
        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_6(self):
        node_id = "site6"
        ham_tensor = self.hamiltonian.tensors[node_id]
        cache_tensor = self.cache.get_entry("site0",node_id)
        ref_tensor = tensordot(ham_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.cache.get_entry("site7",node_id)
        ref_tensor = tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        ref_tensor = transpose(ref_tensor, axes=(3,5,0,2,4,1))

        found_tensor = contract_all_except_node(node_id,
                                                     self.state,
                                                     self.hamiltonian,
                                                     self.cache)
        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_1(self):
        node_id = "site1"
        ham_tensor = self.hamiltonian.tensors[node_id]
        cache_tensor = self.cache.get_entry("site0",node_id)
        ref_tensor = tensordot(ham_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.cache.get_entry("site2",node_id)
        ref_tensor = tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.cache.get_entry("site3",node_id)
        ref_tensor = tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        ref_tensor = transpose(ref_tensor, axes=(3,7,5,0,2,6,4,1))

        found_tensor = contract_all_except_node(node_id,
                                                     self.state,
                                                     self.hamiltonian,
                                                     self.cache)
        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_3(self):
        node_id = "site3"
        ham_tensor = self.hamiltonian.tensors[node_id]
        cache_tensor = self.cache.get_entry("site1",node_id)
        ref_tensor = tensordot(ham_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.cache.get_entry("site4",node_id)
        ref_tensor = tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.cache.get_entry("site5",node_id)
        ref_tensor = tensordot(ref_tensor,
                                  cache_tensor,
                                  axes=(0,1))
        ref_tensor = transpose(ref_tensor, axes=(3,5,7,0,2,4,6,1))

        found_tensor = contract_all_except_node(node_id,
                                                     self.state,
                                                     self.hamiltonian,
                                                     self.cache)
        self.assertTrue(allclose(ref_tensor,found_tensor))

    def test_get_effective_hamiltonian_4(self):
        node_id = "site4"
        ref_matrix = contract_all_except_node(node_id,
                                              self.state,
                                              self.hamiltonian,
                                              self.cache)
        ref_matrix = reshape(ref_matrix,(4,4))

        found_matrix = get_effective_single_site_hamiltonian(node_id,
                                                            self.state,
                                                            self.hamiltonian,
                                                            self.cache)
        self.assertTrue(allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_5(self):
        node_id = "site5"
        ref_matrix = contract_all_except_node(node_id,
                                              self.state,
                                              self.hamiltonian,
                                              self.cache)
        ref_matrix = reshape(ref_matrix,(4,4))

        found_matrix = get_effective_single_site_hamiltonian(node_id,
                                                            self.state,
                                                            self.hamiltonian,
                                                            self.cache)
        self.assertTrue(allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_2(self):
        node_id = "site2"
        ref_matrix = contract_all_except_node(node_id,
                                              self.state,
                                              self.hamiltonian,
                                              self.cache)
        ref_matrix = reshape(ref_matrix,(4,4))

        found_matrix = get_effective_single_site_hamiltonian(node_id,
                                                            self.state,
                                                            self.hamiltonian,
                                                            self.cache)
        self.assertTrue(allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_7(self):
        node_id = "site7"
        ref_matrix = contract_all_except_node(node_id,
                                              self.state,
                                              self.hamiltonian,
                                              self.cache)
        ref_matrix = reshape(ref_matrix,(4,4))

        found_matrix = get_effective_single_site_hamiltonian(node_id,
                                                            self.state,
                                                            self.hamiltonian,
                                                            self.cache)
        self.assertTrue(allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_0(self):
        node_id = "site0"
        ref_matrix = contract_all_except_node(node_id,
                                              self.state,
                                              self.hamiltonian,
                                              self.cache)
        ref_matrix = reshape(ref_matrix,(8,8))

        found_matrix = get_effective_single_site_hamiltonian(node_id,
                                                            self.state,
                                                            self.hamiltonian,
                                                            self.cache)
        self.assertTrue(allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_6(self):
        node_id = "site6"
        ref_matrix = contract_all_except_node(node_id,
                                              self.state,
                                              self.hamiltonian,
                                              self.cache)
        ref_matrix = reshape(ref_matrix,(8,8))

        found_matrix = get_effective_single_site_hamiltonian(node_id,
                                                            self.state,
                                                            self.hamiltonian,
                                                            self.cache)
        self.assertTrue(allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_1(self):
        node_id = "site1"
        ref_matrix = contract_all_except_node(node_id,
                                              self.state,
                                              self.hamiltonian,
                                              self.cache)
        ref_matrix = reshape(ref_matrix,(16,16))

        found_matrix = get_effective_single_site_hamiltonian(node_id,
                                                            self.state,
                                                            self.hamiltonian,
                                                            self.cache)
        self.assertTrue(allclose(ref_matrix,found_matrix))

    def test_get_effective_hamiltonian_3(self):
        node_id = "site3"
        ref_matrix = contract_all_except_node(node_id,
                                              self.state,
                                              self.hamiltonian,
                                              self.cache)
        ref_matrix = reshape(ref_matrix,(16,16))

        found_matrix = get_effective_single_site_hamiltonian(node_id,
                                                            self.state,
                                                            self.hamiltonian,
                                                            self.cache)
        self.assertTrue(allclose(ref_matrix,found_matrix))

if __name__ == "__main__":
    unittest.main()
