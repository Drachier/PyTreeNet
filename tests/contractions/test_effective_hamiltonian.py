import unittest

from numpy import tensordot, transpose, allclose, reshape

from pytreenet.core.graph_node import GraphNode
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno,
                                                   RandomTTNSMode)
from pytreenet.random.random_node import random_tensor_node
from pytreenet.random.random_matrices import crandn
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.contractions.sandwich_caching import SandwichCache

from pytreenet.contractions.effective_hamiltonians import (find_tensor_leg_permutation,
                                                           contract_all_except_node,
                                                           get_effective_single_site_hamiltonian_nodes,
                                                           get_effective_single_site_hamiltonian, get_effective_two_site_hamiltonian)

class TestFindTensorLegPermutation(unittest.TestCase):
    """
    Test the find_tensor_leg_permutation function for a variety of cases.

    Notably it considers the cases were the children of the state node and the
    Hamiltonian node are in the same order and in different order.
    """

    def test_for_leaf(self):
        """
        Test the permutation finding for a leaf node.
        """
        state_node, _ = random_tensor_node((6,3))
        hamiltonian_node, _ = random_tensor_node((6,2,3))
        parent_id = "parent"
        parent_leg = 0
        state_node.open_leg_to_parent(parent_id, parent_leg)
        hamiltonian_node.open_leg_to_parent(parent_id, parent_leg)
        permutation = find_tensor_leg_permutation(state_node, hamiltonian_node)
        correct_perm = (3,0,2,1)
        self.assertEqual(permutation, correct_perm)

    def test_for_root_sorted(self):
        """
        Test the permutation finding for a root, where the children of state
        and Hamiltonian are in the same order.
        """
        state_node, _ = random_tensor_node((4,5,3))
        hamiltonian_node, _ = random_tensor_node((4,5,2,3))
        c_dict = {"c1":0, "c2":1}
        state_node.open_legs_to_children(c_dict)
        hamiltonian_node.open_legs_to_children(c_dict)
        permutation = find_tensor_leg_permutation(state_node, hamiltonian_node)
        correct_perm = (3,5,0,2,4,1)
        self.assertEqual(permutation, correct_perm)

    def test_for_root_unsorted(self):
        """
        Test the permutation finding for a root, where the children of state
        and Hamiltonian are in different order.
        """
        state_node, _ = random_tensor_node((4,5,3))
        hamiltonian_node, _ = random_tensor_node((4,5,2,3))
        state_node.open_legs_to_children({"c1":0, "c2":1})
        hamiltonian_node.open_legs_to_children({"c2":1, "c1":0})
        permutation = find_tensor_leg_permutation(state_node, hamiltonian_node)
        correct_perm = (5,3,0,4,2,1)
        self.assertEqual(permutation, correct_perm)

    def test_for_node_sorted(self):
        """
        Test the permutation finding for a node that has children and a parent.
        The children of the state node and the Hamiltonian node are in the same
        order.
        """
        state_node, _ = random_tensor_node((6,4,5,3))
        hamiltonian_node, _ = random_tensor_node((6,4,5,2,3))
        state_node.open_leg_to_parent("parent", 0)
        hamiltonian_node.open_leg_to_parent("parent", 0)
        state_node.open_legs_to_children({"c1":1, "c2":2})
        hamiltonian_node.open_legs_to_children({"c1":1, "c2":2})
        permutation = find_tensor_leg_permutation(state_node, hamiltonian_node)
        correct_perm = (3,5,7,0,2,4,6,1)
        self.assertEqual(permutation, correct_perm)

    def test_for_node_unsorted(self):
        """
        Test the permutation finding for a node that has children and a parent.
        The children of the state node and the Hamiltonian node are in different
        order.
        """
        state_node, _ = random_tensor_node((6,4,5,3))
        hamiltonian_node, _ = random_tensor_node((6,4,5,2,3))
        state_node.open_leg_to_parent("parent", 0)
        hamiltonian_node.open_leg_to_parent("parent", 0)
        state_node.open_legs_to_children({"c1":1, "c2":2})
        hamiltonian_node.open_legs_to_children({"c2":2, "c1":1})
        permutation = find_tensor_leg_permutation(state_node, hamiltonian_node)
        correct_perm = (3,7,5,0,2,6,4,1)
        self.assertEqual(permutation, correct_perm)

class TestContractAllExceptNode(unittest.TestCase):
    """
    Test the contract_all_except_node function for a variety of cases.

    Notably it considers the cases were the children of the state node and the
    Hamiltonian node are in the same order and in different order.
    """

    def test_on_leaf(self):
        """
        Test the contraction of all tensors except the leaf node.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((6,3),identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((6,2,3),
                                                                  identifier=node_id)
        parent_id = "parent"
        parent_leg = 0
        state_node.open_leg_to_parent(parent_id, parent_leg)
        hamiltonian_node.open_leg_to_parent(parent_id, parent_leg)
        cache = PartialTreeCachDict()
        parent_env = crandn(6,6,6)
        cache.add_entry(parent_id, node_id, parent_env)
        contracted_tensor = contract_all_except_node(state_node,
                                                     hamiltonian_node,
                                                     hamiltonian_tensor,
                                                     cache)
        correct_tensor = tensordot(hamiltonian_tensor,
                                   parent_env,
                                   axes=(0,1))
        correct_tensor = correct_tensor.transpose((3,0,2,1))
        self.assertEqual((6,2,6,3), contracted_tensor.shape)
        self.assertTrue(allclose(correct_tensor, contracted_tensor))

    def test_on_root_sorted(self):
        """
        Test the contraction of all tensors except the root node.
        The children of the state node and the Hamiltonian node are in the same
        order.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((4,5,3), identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((4,5,2,3),
                                                                  identifier=node_id)
        c_dict = {"c1":0, "c2":1}
        state_node.open_legs_to_children(c_dict)
        hamiltonian_node.open_legs_to_children(c_dict)
        cache = PartialTreeCachDict()
        cache.add_entry("c1", node_id, crandn(4,4,4))
        cache.add_entry("c2", node_id, crandn(5,5,5))
        contracted_tensor = contract_all_except_node(state_node,
                                                     hamiltonian_node,
                                                     hamiltonian_tensor,
                                                     cache)
        # Correct Contraction
        correct_tensor = tensordot(hamiltonian_tensor,
                                   cache.get_entry("c1",node_id),
                                   axes=(0,1))
        correct_tensor = tensordot(correct_tensor,
                                   cache.get_entry("c2",node_id),
                                   axes=(0,1))
        correct_tensor = correct_tensor.transpose((3,5,0,2,4,1))
        self.assertEqual((4,5,2,4,5,3), contracted_tensor.shape)
        self.assertTrue(allclose(correct_tensor, contracted_tensor))

    def test_on_root_unsorted(self):
        """
        Test the contraction of all tensors except the root node.
        The children of the state node and the Hamiltonian node are in different
        order.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((4,5,3), identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((4,5,2,3),
                                                                  identifier=node_id)
        state_node.open_legs_to_children({"c1":0, "c2":1})
        hamiltonian_node.open_legs_to_children({"c2":1, "c1":0})
        hamiltonian_tensor = hamiltonian_tensor.transpose((1,0,2,3)) # Bring it to the correct order
        cache = PartialTreeCachDict()
        cache.add_entry("c1", node_id, crandn(4,4,4))
        cache.add_entry("c2", node_id, crandn(5,5,5))
        contracted_tensor = contract_all_except_node(state_node,
                                                     hamiltonian_node,
                                                     hamiltonian_tensor,
                                                     cache)
        # Correct Contraction
        correct_tensor = tensordot(hamiltonian_tensor,
                                   cache.get_entry("c2",node_id),
                                   axes=(0,1))
        correct_tensor = tensordot(correct_tensor,
                                   cache.get_entry("c1",node_id),
                                   axes=(0,1))
        correct_tensor = correct_tensor.transpose((5,3,0,4,2,1))
        self.assertEqual((4,5,2,4,5,3), contracted_tensor.shape)
        self.assertTrue(allclose(correct_tensor, contracted_tensor))

    def test_on_node_sorted(self):
        """
        Test the contraction of all tensors except a node that has children and
        a parent. The children of the state node and the Hamiltonian node are in
        the same order.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((6,4,5,3), identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((6,4,5,2,3), identifier=node_id)
        state_node.open_leg_to_parent("parent", 0)
        hamiltonian_node.open_leg_to_parent("parent", 0)
        state_node.open_legs_to_children({"c1":1, "c2":2})
        hamiltonian_node.open_legs_to_children({"c1":1, "c2":2})
        cache = PartialTreeCachDict()
        cache.add_entry("c1", node_id, crandn(4,4,4))
        cache.add_entry("c2", node_id, crandn(5,5,5))
        cache.add_entry("parent", node_id, crandn(6,6,6))
        contracted_tensor = contract_all_except_node(state_node,
                                                     hamiltonian_node,
                                                     hamiltonian_tensor,
                                                     cache)
        # Correct Contraction
        correct_tensor = tensordot(hamiltonian_tensor,
                                   cache.get_entry("parent",node_id),
                                   axes=(0,1))
        correct_tensor = tensordot(correct_tensor,
                                   cache.get_entry("c1",node_id),
                                   axes=(0,1))
        correct_tensor = tensordot(correct_tensor,
                                   cache.get_entry("c2",node_id),
                                   axes=(0,1))
        correct_tensor = correct_tensor.transpose((3,5,7,0,2,4,6,1))
        self.assertEqual((6,4,5,2,6,4,5,3), contracted_tensor.shape)
        self.assertTrue(allclose(correct_tensor, contracted_tensor))

    def test_on_node_unsorted(self):
        """
        Test the contraction of all tensors except a node that has children and
        a parent. The children of the state node and the Hamiltonian node are in
        different order.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((6,4,5,3),identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((6,4,5,2,3),identifier=node_id)
        state_node.open_leg_to_parent("parent", 0)
        hamiltonian_node.open_leg_to_parent("parent", 0)
        state_node.open_legs_to_children({"c1":1, "c2":2})
        hamiltonian_node.open_legs_to_children({"c2":2, "c1":1})
        ## Bring it to the correct order
        hamiltonian_tensor = hamiltonian_tensor.transpose((0,2,1,3,4))
        cache = PartialTreeCachDict()
        cache.add_entry("c1", node_id, crandn(4,4,4))
        cache.add_entry("c2", node_id, crandn(5,5,5))
        cache.add_entry("parent", node_id, crandn(6,6,6))
        contracted_tensor = contract_all_except_node(state_node,
                                                     hamiltonian_node,
                                                     hamiltonian_tensor,
                                                     cache)
        # Correct Contraction
        correct_tensor = tensordot(hamiltonian_tensor,
                                   cache.get_entry("parent",node_id),
                                   axes=(0,1))
        correct_tensor = tensordot(correct_tensor,
                                   cache.get_entry("c2",node_id),
                                   axes=(0,1))
        correct_tensor = tensordot(correct_tensor,
                                   cache.get_entry("c1",node_id),
                                   axes=(0,1))
        correct_tensor = correct_tensor.transpose((3,7,5,0,2,6,4,1))
        self.assertEqual((6,4,5,2,6,4,5,3), contracted_tensor.shape)
        self.assertTrue(allclose(correct_tensor, contracted_tensor))

class TestGetEffectiveSingleSiteHamiltonianNodes(unittest.TestCase):
    """
    Tests the get_effective_single_site_hamiltonian_nodes function for a variety
    of cases.

    Notably it considers the cases were the children of the state node and the
    Hamiltonian node are in the same order and in different order.
    """

    def test_on_leaf(self):
        """
        Test the finding of the effective single site Hamiltonian for a leaf.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((6,3),identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((6,2,3),
                                                                  identifier=node_id)
        parent_id = "parent"
        parent_leg = 0
        state_node.open_leg_to_parent(parent_id, parent_leg)
        hamiltonian_node.open_leg_to_parent(parent_id, parent_leg)
        cache = PartialTreeCachDict()
        parent_env = crandn(6,6,6)
        cache.add_entry(parent_id, node_id, parent_env)
        h_eff = get_effective_single_site_hamiltonian_nodes(state_node,
                                                            hamiltonian_node,
                                                            hamiltonian_tensor,
                                                            cache)
        self.assertEqual((6*2,6*3), h_eff.shape)
        correct_tensor = contract_all_except_node(state_node,
                                                  hamiltonian_node,
                                                  hamiltonian_tensor,
                                                  cache)
        correct_tensor = correct_tensor.reshape((6*2,6*3))
        self.assertTrue(allclose(correct_tensor, h_eff))

    def test_on_root_sorted(self):
        """
        Test the finding of the effective single site Hamiltonian for a root.
        The children of the state node and the Hamiltonian node are in the same
        order.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((4,5,3), identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((4,5,2,3),
                                                                  identifier=node_id)
        c_dict = {"c1":0, "c2":1}
        state_node.open_legs_to_children(c_dict)
        hamiltonian_node.open_legs_to_children(c_dict)
        cache = PartialTreeCachDict()
        cache.add_entry("c1", node_id, crandn(4,4,4))
        cache.add_entry("c2", node_id, crandn(5,5,5))
        h_eff = get_effective_single_site_hamiltonian_nodes(state_node,
                                                            hamiltonian_node,
                                                            hamiltonian_tensor,
                                                            cache)
        self.assertEqual((4*5*2,4*5*3), h_eff.shape)
        correct_tensor = contract_all_except_node(state_node,
                                                  hamiltonian_node,
                                                  hamiltonian_tensor,
                                                  cache)
        correct_tensor = correct_tensor.reshape((4*5*2,4*5*3))
        self.assertTrue(allclose(correct_tensor, h_eff))

    def test_on_root_unsorted(self):
        """
        Test the finding of the effective single site Hamiltonian for a root.
        The children of the state node and the Hamiltonian node are in different
        order.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((4,5,3), identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((4,5,2,3),
                                                                  identifier=node_id)
        state_node.open_legs_to_children({"c1":0, "c2":1})
        hamiltonian_node.open_legs_to_children({"c2":1, "c1":0})
        hamiltonian_tensor = hamiltonian_tensor.transpose((1,0,2,3)) # Bring it to the correct order
        cache = PartialTreeCachDict()
        cache.add_entry("c1", node_id, crandn(4,4,4))
        cache.add_entry("c2", node_id, crandn(5,5,5))
        h_eff = get_effective_single_site_hamiltonian_nodes(state_node,
                                                            hamiltonian_node,
                                                            hamiltonian_tensor,
                                                            cache)
        self.assertEqual((4*5*2,4*5*3), h_eff.shape)
        correct_tensor = contract_all_except_node(state_node,
                                                  hamiltonian_node,
                                                  hamiltonian_tensor,
                                                  cache)
        correct_tensor = correct_tensor.reshape((4*5*2,4*5*3))
        self.assertTrue(allclose(correct_tensor, h_eff))

    def test_on_node_sorted(self):
        """
        Test the finding of the effective single site Hamiltonian for a node that
        has children and a parent. The children of the state node and the
        Hamiltonian node are in the same order.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((6,4,5,3), identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((6,4,5,2,3), identifier=node_id)
        state_node.open_leg_to_parent("parent", 0)
        hamiltonian_node.open_leg_to_parent("parent", 0)
        state_node.open_legs_to_children({"c1":1, "c2":2})
        hamiltonian_node.open_legs_to_children({"c1":1, "c2":2})
        cache = PartialTreeCachDict()
        cache.add_entry("c1", node_id, crandn(4,4,4))
        cache.add_entry("c2", node_id, crandn(5,5,5))
        cache.add_entry("parent", node_id, crandn(6,6,6))
        h_eff = get_effective_single_site_hamiltonian_nodes(state_node,
                                                            hamiltonian_node,
                                                            hamiltonian_tensor,
                                                            cache)
        self.assertEqual((6*4*5*2,6*4*5*3), h_eff.shape)
        correct_tensor = contract_all_except_node(state_node,
                                                    hamiltonian_node,
                                                    hamiltonian_tensor,
                                                    cache)
        correct_tensor = correct_tensor.reshape((6*4*5*2,6*4*5*3))
        self.assertTrue(allclose(correct_tensor, h_eff))

    def test_on_node_unsorted(self):
        """
        Test the finding of the effective single site Hamiltonian for a node that
        has children and a parent. The children of the state node and the
        Hamiltonian node are in different order.
        """
        node_id = "node"
        state_node, _ = random_tensor_node((6,4,5,3),identifier=node_id)
        hamiltonian_node, hamiltonian_tensor = random_tensor_node((6,4,5,2,3),identifier=node_id)
        state_node.open_leg_to_parent("parent", 0)
        hamiltonian_node.open_leg_to_parent("parent", 0)
        state_node.open_legs_to_children({"c1":1, "c2":2})
        hamiltonian_node.open_legs_to_children({"c2":2, "c1":1})
        ## Bring it to the correct order
        hamiltonian_tensor = hamiltonian_tensor.transpose((0,2,1,3,4))
        cache = PartialTreeCachDict()
        cache.add_entry("c1", node_id, crandn(4,4,4))
        cache.add_entry("c2", node_id, crandn(5,5,5))
        cache.add_entry("parent", node_id, crandn(6,6,6))
        h_eff = get_effective_single_site_hamiltonian_nodes(state_node,
                                                            hamiltonian_node,
                                                            hamiltonian_tensor,
                                                            cache)
        self.assertEqual((6*4*5*2,6*4*5*3), h_eff.shape)
        correct_tensor = contract_all_except_node(state_node,
                                                  hamiltonian_node,
                                                  hamiltonian_tensor,
                                                  cache)
        correct_tensor = correct_tensor.reshape((6*4*5*2,6*4*5*3))
        self.assertTrue(allclose(correct_tensor, h_eff))                                       

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
        state_node = self.ttns.nodes["root"]
        hamiltonian_node = self.hamiltonian.nodes["root"]
        permutation = find_tensor_leg_permutation(state_node,
                                                    hamiltonian_node)
        correct_perm = tuple([3,5,0,2,4,1])
        self.assertEqual(permutation, correct_perm)

    def test_find_tensor_leg_permutation_diff(self):
        """
        Test the permutation finding for a node that has children and a parent.
        The children of the state node and the Hamiltonian node are in different
        order.
        """
        # Move centre to make the children in different order
        self.ttns.canonical_form("c1")
        self.ttns.canonical_form("c2")
        state_node = self.ttns.nodes["root"]
        hamiltonian_node = self.hamiltonian.nodes["root"]
        print(state_node.children, hamiltonian_node.children)
        permutation = find_tensor_leg_permutation(state_node,
                                                    hamiltonian_node)
        correct_perm = tuple([5,3,0,4,2,1])
        self.assertEqual(permutation, correct_perm)

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

    def test_get_effective_two_site_hamiltonian_root_c2(self):
        # Compute Reference
        ref_tensor = tensordot(self.cache.get_entry("c1", "root"),
                                  self.hamiltonian.tensors["root"],
                                  axes=(1,0))
        ref_tensor = tensordot(ref_tensor, self.hamiltonian.tensors["c2"],
                               axes=(2,0))
        ref_tensor = transpose(ref_tensor, axes=(1,4,2,0,5,3))
        ref_tensor = reshape(ref_tensor, (40,40))
        self.ttns.contract_nodes("c2", "root", 'TwoSite_c2_contr_root')
        found_tensor = get_effective_two_site_hamiltonian("c2",
                                                            "root",
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
        self.state, self.hamiltonian = big_ttns_and_ttno(mode=RandomTTNSMode.DIFFVIRT)
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
        

if __name__ == "__main__":
    unittest.main()
