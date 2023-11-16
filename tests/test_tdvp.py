
import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn

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
        partial_tree_cache = ptn.PartialTreeChachDict()
        c2_cache = ptn.PartialTreeCache.for_leaf("c2", ref_tdvp.state,
                                                 ref_tdvp.hamiltonian)
        partial_tree_cache.add_entry("c2", "root", c2_cache)
        root_cache = ptn.PartialTreeCache.with_existing_cache("root", "c1",
                                                              partial_tree_cache,
                                                              ref_tdvp.state,
                                                              ref_tdvp.hamiltonian)
        partial_tree_cache.add_entry("root", "c1", root_cache)

        for ids, cache in partial_tree_cache.items():
            found_cache = self.tdvp.partial_tree_cache.get_entry(ids[0], ids[1])
            self.assertTrue(cache.is_close(found_cache))

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
        c1_cache = ptn.PartialTreeCache.for_leaf("c1", self.tdvp.state,
                                                  self.tdvp.hamiltonian)
        self.tdvp.partial_tree_cache.add_entry("c1", "root", c1_cache)
        root_to_c2_cache = ptn.PartialTreeCache.with_existing_cache("root",
                                                                    "c2",
                                                                    self.tdvp.partial_tree_cache,
                                                                    self.tdvp.state,
                                                                    self.tdvp.hamiltonian)
        self.tdvp.partial_tree_cache.add_entry("root", "c2", root_to_c2_cache)

    def test_contract_all_except_node_c1(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["c1"]
        cache_tensor = self.tdvp.partial_tree_cache.get_cached_tensor("root", "c1")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,0,2,1])
        found_tensor = self.tdvp._contract_all_except_node("c1")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_c2(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["c2"]
        cache_tensor = self.tdvp.partial_tree_cache.get_cached_tensor("root", "c2")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,0,2,1])

        found_tensor = self.tdvp._contract_all_except_node("c2")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_all_except_node_root(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["root"]
        cache_tensor = self.tdvp.partial_tree_cache.get_cached_tensor("c1", "root")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_cached_tensor("c2", "root")
        ref_tensor = np.tensordot(ref_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,5,0,2,4,1])

        found_tensor = self.tdvp._contract_all_except_node("root")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_get_effective_site_hamiltonian_c1(self):
        # Copmute Reference
        ref_tensor = np.tensordot(self.tdvp.partial_tree_cache.get_cached_tensor("root", "c1"),
                                  self.tdvp.hamiltonian.tensors["c1"],
                                  axes=(1,0))
        ref_tensor = np.transpose(ref_tensor, axes=(1,2,0,3))
        ref_tensor = np.reshape(ref_tensor, (15,15))

        found_tensor = self.tdvp._get_effective_site_hamiltonian("c1")
        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_get_effective_site_hamiltonian_c2(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["c2"]
        cache_tensor = self.tdvp.partial_tree_cache.get_cached_tensor("root", "c2")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,0,2,1])
        ref_tensor = np.reshape(ref_tensor, (16,16))

        found_tensor = self.tdvp._get_effective_site_hamiltonian("c2")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_get_effective_site_hamiltonian_root(self):
        # Compute Reference
        ham_tensor = self.tdvp.hamiltonian.tensors["root"]
        cache_tensor = self.tdvp.partial_tree_cache.get_cached_tensor("c1", "root")
        ref_tensor = np.tensordot(ham_tensor, cache_tensor,
                                  axes=(0,1))
        cache_tensor = self.tdvp.partial_tree_cache.get_cached_tensor("c2", "root")
        ref_tensor = np.tensordot(ref_tensor, cache_tensor,
                                  axes=(0,1))
        ref_tensor = np.transpose(ref_tensor, axes=[3,5,0,2,4,1])
        ref_tensor = np.reshape(ref_tensor, (40,40))

        found_tensor = self.tdvp._get_effective_site_hamiltonian("root")

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_update_cache_after_split_c1_to_root(self):
        node_id = "c1"
        q_legs = ptn.LegSpecification(None,[],[1])
        r_legs = ptn.LegSpecification("root",[],[])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id,"root"))

        # Compute Reference
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(2,1))

        self.tdvp._update_cache_after_split(node_id, "root")
        found_tensor = self.tdvp.partial_tree_cache.get_cached_tensor(node_id,"root")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_update_cache_after_split_c2_to_root(self):
        node_id = "c2"
        q_legs = ptn.LegSpecification(None,[],[1])
        r_legs = ptn.LegSpecification("root",[],[])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id,"root"))

        # Compute Reference
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(1,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(2,1))

        self.tdvp._update_cache_after_split(node_id, "root")
        found_tensor = self.tdvp.partial_tree_cache.get_cached_tensor(node_id,"root")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_update_cache_after_split_root_to_c1(self):
        node_id = "root"
        q_legs = ptn.LegSpecification(None,["c2"],[2])
        r_legs = ptn.LegSpecification(None,["c1"],[])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id, "c1"))

        # Compute Reference
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(2,3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(4,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.partial_tree_cache.get_cached_tensor("c2",node_id),
                                  axes=([1,3,5],[0,1,2]))

        self.tdvp._update_cache_after_split(node_id,"c1")
        found_tensor = self.tdvp.partial_tree_cache.get_cached_tensor(node_id,"c1")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_update_cache_after_split_root_to_c2(self):
        node_id = "root"
        q_legs = ptn.LegSpecification(None,["c1"],[2])
        r_legs = ptn.LegSpecification(None,["c2"],[])
        self.tdvp.state.split_node_qr(node_id, q_legs, r_legs,
                                      q_identifier=node_id,
                                      r_identifier=self.tdvp.create_link_id(node_id, "c2"))
        # Compute Reference
        ref_tensor = np.tensordot(self.tdvp.state.tensors[node_id],
                                  self.tdvp.hamiltonian.tensors[node_id],
                                  axes=(2,3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.tdvp.state.tensors[node_id].conj(),
                                  axes=(4,2))
        cached_node = self.tdvp.partial_tree_cache.get_cached_tensor("c1",node_id)
        # Note the link tensor is the first child of root not c1!
        ref_tensor = np.tensordot(ref_tensor,
                                  cached_node,
                                  axes=([1,3,5],[0,1,2]))

        self.tdvp._update_cache_after_split(node_id,"c2")
        found_tensor = self.tdvp.partial_tree_cache.get_cached_tensor(node_id,"c2")

        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_split_updated_site_c1(self):
        node_id = "c1"
        q_legs = ptn.LegSpecification(None,[],[1])
        r_legs = ptn.LegSpecification("root",[],[])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(node_id, q_legs, r_legs,
                                q_identifier=node_id,
                                r_identifier=self.tdvp.create_link_id(node_id, "root"))
        ref_old_cache = deepcopy(self.tdvp.partial_tree_cache.get_cached_tensor(node_id,"root"))

        self.tdvp._split_updated_site(node_id,"root")
        self.assertTrue(np.allclose(ref_state.tensors[node_id],
                                    self.tdvp.state.tensors[node_id]))
        try: # Shape change due to QR-decomp
            self.assertFalse(np.allclose(ref_old_cache,
                                         self.tdvp.partial_tree_cache.get_cached_tensor(node_id,"root")))
        except ValueError:
            pass
        link_id = self.tdvp.create_link_id(node_id,"root")
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_split_updated_site_c2(self):
        node_id = "c2"
        q_legs = ptn.LegSpecification(None,[],[1])
        r_legs = ptn.LegSpecification("root",[],[])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(node_id, q_legs, r_legs,
                                q_identifier=node_id,
                                r_identifier=self.tdvp.create_link_id(node_id, "root"))

        self.tdvp._split_updated_site(node_id,"root")
        self.assertTrue(np.allclose(ref_state.tensors[node_id],
                                    self.tdvp.state.tensors[node_id]))
        link_id = self.tdvp.create_link_id(node_id,"root")
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_split_updated_site_root_to_c1(self):
        node_id = "c1"
        root_id = "root"
        q_legs = ptn.LegSpecification(None,["c2"],[2])
        r_legs = ptn.LegSpecification(None,[node_id],[])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(root_id, q_legs, r_legs,
                                q_identifier=root_id,
                                r_identifier=self.tdvp.create_link_id(root_id, node_id))

        self.tdvp._split_updated_site(root_id, node_id)
        self.assertTrue(np.allclose(ref_state.tensors[root_id],
                                    self.tdvp.state.tensors[root_id]))
        link_id = self.tdvp.create_link_id(root_id,node_id)
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_split_updated_site_root_to_c2(self):
        node_id = "c2"
        root_id = "root"
        q_legs = ptn.LegSpecification(None,["c1"],[2])
        r_legs = ptn.LegSpecification(None,[node_id],[])
        ref_state = deepcopy(self.tdvp.state)
        ref_state.split_node_qr(root_id, q_legs, r_legs,
                                q_identifier=root_id,
                                r_identifier=self.tdvp.create_link_id(root_id, node_id))

        self.tdvp._split_updated_site(root_id, node_id)
        self.assertTrue(np.allclose(ref_state.tensors[root_id],
                                    self.tdvp.state.tensors[root_id]))
        link_id = self.tdvp.create_link_id(root_id,node_id)
        self.assertTrue(link_id in self.tdvp.state)
        self.assertTrue(np.allclose(ref_state.tensors[link_id],
                                    self.tdvp.state.tensors[link_id]))

    def test_split_updated_site_exception(self):
        self.assertRaises(ptn.NoConnectionException,
                          self.tdvp._split_updated_site,
                          "c1", "c2")

    def test_get_effective_link_hamiltonian_c1_to_root(self):
        root_id = "root"
        node_id = "c1"
        self.tdvp._split_updated_site(node_id, root_id)
        cache_c1 = self.tdvp.partial_tree_cache.get_cached_tensor(root_id,node_id)
        cache_root = self.tdvp.partial_tree_cache.get_cached_tensor(node_id,root_id)
        ref_tensor = np.tensordot(cache_c1,cache_root,
                                  axes=(1,1))
        ref_tensor = np.transpose(ref_tensor, axes=[1,3,0,2])
        ref_tensor = np.reshape(ref_tensor, (15,15))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(node_id,root_id)

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_get_effective_link_hamiltonian_c2_to_root(self):
        root_id = "root"
        node_id = "c2"
        self.tdvp._split_updated_site(node_id, root_id)
        cache_c1 = self.tdvp.partial_tree_cache.get_cached_tensor(root_id,node_id)
        cache_root = self.tdvp.partial_tree_cache.get_cached_tensor(node_id,root_id)
        ref_tensor = np.tensordot(cache_c1,cache_root,
                                  axes=(1,1))
        ref_tensor = np.transpose(ref_tensor, axes=[1,3,0,2])
        ref_tensor = np.reshape(ref_tensor, (16,16))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(node_id,root_id)

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_get_effective_link_hamiltonian_root_to_c1(self):
        root_id = "root"
        node_id = "c1"
        self.tdvp._split_updated_site(root_id, node_id)
        cache_c1 = self.tdvp.partial_tree_cache.get_cached_tensor(node_id,root_id)
        cache_root = self.tdvp.partial_tree_cache.get_cached_tensor(root_id,node_id)
        ref_tensor = np.tensordot(cache_root,cache_c1,
                                  axes=(1,1))
        ref_tensor = np.transpose(ref_tensor, axes=[1,3,0,2])
        ref_tensor = np.reshape(ref_tensor, (25,25))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(root_id,node_id)

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_get_effective_link_hamiltonian_root_to_c2(self):
        root_id = "root"
        node_id = "c2"
        self.tdvp._split_updated_site(root_id, node_id)
        cache_c1 = self.tdvp.partial_tree_cache.get_cached_tensor(node_id,root_id)
        cache_root = self.tdvp.partial_tree_cache.get_cached_tensor(root_id,node_id)
        ref_tensor = np.tensordot(cache_root,cache_c1,
                                  axes=(1,1))
        ref_tensor = np.transpose(ref_tensor, axes=[1,3,0,2])
        ref_tensor = np.reshape(ref_tensor, (16,16))

        found_tensor = self.tdvp._get_effective_link_hamiltonian(root_id,node_id)

        self.assertTrue(np.allclose(ref_tensor,found_tensor))

if __name__ == "__main__":
    unittest.main()
