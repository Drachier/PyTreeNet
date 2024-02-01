import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn


class TestTreeTensorNetworkBasics(unittest.TestCase):

    def setUp(self):
        self.tensortree = ptn.TreeTensorNetwork()
        self.node1, self.tensor1 = ptn.random_tensor_node((2, 3, 4, 5), identifier="orig_root")
        self.node2, self.tensor2 = ptn.random_tensor_node((2, 3), identifier="child1")
        self.node3, self.tensor3 = ptn.random_tensor_node((2, 3, 4, 5), identifier="child2")
        self.node4, self.tensor4 = ptn.random_tensor_node((2, 3, 4, 5), identifier="new_root")

    def test_add_root(self):
        self.assertEqual(self.tensortree.root_id, None)
        self.assertEqual(self.tensortree.nodes, {})

        self.tensortree.add_root(self.node1, self.tensor1)

        self.assertEqual(self.tensortree.root_id, "orig_root")
        self.assertEqual(len(self.tensortree.nodes), 1)
        self.assertEqual(len(self.tensortree.tensors), 1)

    def test_add_child_to_parent(self):
        self.tensortree.add_root(self.node1, self.tensor1)

        # Depth 2
        self.tensortree.add_child_to_parent(self.node2, self.tensor2, 1, "orig_root", 1)
        self.assertEqual(len(self.tensortree.nodes), 2)
        permutation = self.tensortree.nodes["child1"].leg_permutation

        transposed_tensor = self.tensor2.transpose(permutation)
        self.assertTrue(np.allclose(self.tensortree.tensors["child1"], transposed_tensor))
        self.assertEqual(self.tensortree.nodes["child1"].parent, "orig_root")
        self.assertEqual(self.tensortree.nodes["child1"].children, [])
        self.assertEqual(self.tensortree.nodes["orig_root"].children, ["child1"])

        # Depth 3
        self.tensortree.add_child_to_parent(self.node3, self.tensor3, 0, "child1", 1)
        self.assertEqual(len(self.tensortree.nodes), 3)
        permutation = self.tensortree.nodes["child2"].leg_permutation
        transposed_tensor = self.tensor3.transpose(permutation)
        self.assertTrue(np.allclose(self.tensortree.tensors["child2"], transposed_tensor))
        self.assertEqual(self.tensortree.nodes["child2"].parent, "child1")
        self.assertEqual(self.tensortree.nodes["child2"].children, [])
        self.assertEqual(self.tensortree.nodes["child1"].children, ["child2"])

    def test_add_parent_to_root(self):
        # Setup
        self.tensortree.add_root(self.node1, self.tensor1)
        self.tensortree.add_child_to_parent(self.node2, self.tensor2, 1, "orig_root", 1)
        self.tensortree.add_child_to_parent(self.node3, self.tensor3, 0, "child1", 1)

        self.tensortree.add_parent_to_root(1, self.node4, self.tensor4, 0)
        self.assertEqual(self.tensortree.root_id, "new_root")
        self.assertEqual(len(self.tensortree.nodes), 4)
        permutation = self.tensortree.nodes["new_root"].leg_permutation
        transposed_tensor = self.tensor4.transpose(permutation)
        self.assertTrue(np.allclose(self.tensortree.tensors["new_root"], transposed_tensor))
        self.assertEqual(self.tensortree.nodes["new_root"].parent, None)
        self.assertEqual(self.tensortree.nodes["new_root"].children, ["orig_root"])
        self.assertEqual(self.tensortree.nodes["orig_root"].parent, "new_root")

class TestTreeTensorNetworkSimple(unittest.TestCase):

    def setUp(self):
        self.tensortree = ptn.random_small_ttns()

    def test_equality_ttn_should_be_equal_to_itself(self):
        ref_tree = deepcopy(self.tensortree)
        self.assertEqual(ref_tree,self.tensortree)
    
    # TODO: Test other situations

    def test_conjugate(self):
        ref_ttn = deepcopy(self.tensortree)
        reference_result = ref_ttn.completely_contract_tree()
        reference_result = reference_result.tensors[reference_result.root_id]
        reference_result = reference_result.conj()

        found_result = self.tensortree.conjugate()
        found_result = found_result.completely_contract_tree()
        found_result = found_result.tensors[found_result.root_id]

        self.assertTrue(np.allclose(reference_result, found_result))

    def test_conjugate_keep_original(self):
        ref_ttn = deepcopy(self.tensortree)
        _ = self.tensortree.conjugate()
        # The original TTN should be unchanged
        for node_id, tensor in self.tensortree.tensors.items():
            self.assertTrue(np.allclose(ref_ttn.tensors[node_id], tensor))

    def test_legs_before_combination_root_c1(self):
        found = self.tensortree.legs_before_combination("root", "c1")
        correct_root = ptn.LegSpecification(None,["c2"], [1], None)
        correct_c1 = ptn.LegSpecification(None, [], [2], None)
        self.assertEqual(found[0], correct_root)
        self.assertEqual(found[1], correct_c1)

    def test_legs_before_combination_c2_root(self):
        found = self.tensortree.legs_before_combination("c2", "root")
        correct_root = ptn.LegSpecification(None,["c1"], [2], None)
        correct_c2 = ptn.LegSpecification(None, [], [1], None)
        self.assertEqual(found[1], correct_root)
        self.assertEqual(found[0], correct_c2)

    def test_contract_nodes_root_c1(self):
        ref_tree = deepcopy(self.tensortree)
        ref_tensor = np.tensordot(ref_tree.tensors["root"],
                                  ref_tree.tensors["c1"],
                                  axes=(0,0))

        self.tensortree.contract_nodes("root", "c1", new_identifier="contr")
        # The new node should be in the TTN
        self.assertTrue("contr" in self.tensortree.nodes)
        # And it should be the root
        self.assertEqual("contr", self.tensortree.root_id)
        found_node, found_tensor = self.tensortree["contr"]
        # The tensors should be the same
        self.assertTrue(np.allclose(ref_tensor, found_tensor))
        # The shape in the new node should be of correct shape
        self.assertEqual((6,2,3), found_node.shape)
        # The old node and tensors should be removeed
        identifiers = ["root", "c1"]
        for identifier in identifiers:
            self.assertFalse(identifier in self.tensortree.nodes)
            self.assertFalse(identifier in self.tensortree.tensors)

    def test_contract_nodes_root_c2(self):
        ref_tree = deepcopy(self.tensortree)
        ref_tensor = np.tensordot(ref_tree.tensors["c2"],
                                  ref_tree.tensors["root"],
                                  axes=(0,1)).transpose(1,0,2)

        self.tensortree.contract_nodes("c2", "root")
        new_id = "c2contrroot"
        # The new node should be in the TTN
        self.assertTrue(new_id in self.tensortree.nodes)
        # And it should be the root
        self.assertEqual(new_id, self.tensortree.root_id)
        found_node, found_tensor = self.tensortree[new_id]
        # The shape in the new node should be of correct shape
        self.assertEqual((5,4,2), found_node.shape)
        # The tensors should be the same
        self.assertTrue(np.allclose(ref_tensor, found_tensor))
        # The old node and tensors should be removeed
        identifiers = ["root", "c2"]
        for identifier in identifiers:
            self.assertFalse(identifier in self.tensortree.nodes)
            self.assertFalse(identifier in self.tensortree.tensors)

    def test_contract_and_split_svd_no_truncation_root_c1(self):
        self.tensortree.contract_nodes("root", "c1", new_identifier="contr")
        ref_tree = deepcopy(self.tensortree)

        root_legs = ptn.LegSpecification(None, ["c2"], [1], None)
        root_legs.is_root = True
        c1_legs = ptn.LegSpecification(None, [], [2], None)
        self.tensortree.split_node_svd("contr", root_legs, c1_legs, "root", "c1"
                                       , max_bond_dim=float("inf"),
                                       rel_tol=float("-inf"), total_tol=float("-inf"))

        # The old node and tensor should be removed
        self.assertFalse("contr" in self.tensortree.nodes)
        self.assertFalse("contr" in self.tensortree.tensors)
        # "root" should be the root again
        self.assertTrue("root", self.tensortree.root_id)

        # There should be a node and a tensor for both
        identifiers = ["root", "c1"]
        for identifier in identifiers:
            self.assertTrue(identifier in self.tensortree.nodes)
            self.assertTrue(identifier in self.tensortree.tensors)

        root_node, root_tensor = self.tensortree["root"]
        c1_node = self.tensortree.nodes["c1"]
        # The children should be correct
        self.assertEqual(["c1", "c2"], root_node.children)
        self.assertTrue(c1_node.is_leaf())
        # The parents should be correct
        self.assertTrue(root_node.is_root())
        self.assertEqual("root", c1_node.parent)
        # Open dimensions should be correct
        self.assertEqual(2, root_node.open_dimension())
        self.assertEqual(3, c1_node.open_dimension())

        # New leg dimension should be equal
        self.assertEqual(root_node.shape[0], c1_node.shape[0])
        # Root node should have expected shape
        self.assertEqual((c1_node.shape[0],6,2), root_node.shape)
        # Root tensor should be an isometry
        identity = np.eye(root_node.shape[0])
        found_identity = np.tensordot(root_tensor.conj(), root_tensor,
                                      axes=((1,2),(1,2)))
        self.assertTrue(np.allclose(identity, found_identity))

        # Since no truncation happend, the two TTN should be the same
        reference = ref_tree.completely_contract_tree().root[1]
        found = self.tensortree.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(reference, found))

    def test_contract_and_split_with_svd_no_truncation_c2_root(self):
        self.tensortree.contract_nodes("c2", "root", new_identifier="contr")
        ref_tree = deepcopy(self.tensortree)

        root_legs = ptn.LegSpecification(None, ["c1"], [2], None)
        root_legs.is_root = True
        c2_legs = ptn.LegSpecification(None, [], [1], None)
        self.tensortree.split_node_svd("contr", c2_legs, root_legs, "c2", "root"
                                       , max_bond_dim=float("inf"),
                                       rel_tol=float("-inf"), total_tol=float("-inf"))

        # The old node and tensor should be removed
        self.assertFalse("contr" in self.tensortree.nodes)
        self.assertFalse("contr" in self.tensortree.tensors)
        # "root" should be the root again
        self.assertTrue("root", self.tensortree.root_id)

        # There should be a node and a tensor for both
        identifiers = ["root", "c1"]
        for identifier in identifiers:
            self.assertTrue(identifier in self.tensortree.nodes)
            self.assertTrue(identifier in self.tensortree.tensors)

        root_node, _ = self.tensortree["root"]
        c2_node, c2_tensor = self.tensortree["c2"]
        # The children should be correct
        self.assertEqual(["c2", "c1"], root_node.children)
        self.assertTrue(c2_node.is_leaf())
        # The parents should be correct
        self.assertTrue(root_node.is_root())
        self.assertEqual("root", c2_node.parent)
        # Open dimensions should be correct
        self.assertEqual(2, root_node.open_dimension())
        self.assertEqual(4, c2_node.open_dimension())

        # New leg dimension should be equal
        self.assertEqual(root_node.shape[0], c2_node.shape[0])
        # Root node should have expected shape
        self.assertEqual((c2_node.shape[0],5,2), root_node.shape)
        # c2 tensor should be an isometry
        identity = np.eye(c2_node.shape[0])
        found_identity = np.tensordot(c2_tensor.conj(), c2_tensor,
                                      axes=((0,),(0,)))
        self.assertTrue(np.allclose(identity, found_identity))

        # Since no truncation happend, the two TTN should be the same
        reference = ref_tree.completely_contract_tree().root[1].transpose(1,0,2)
        found = self.tensortree.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(reference, found))

    def test_move_orthogonalisation_center_None(self):
        self.assertRaises(AssertionError,
                          self.tensortree.move_orthogonalization_center,
                          "root")

    def test_move_center_to_itself(self):
        reference_tree = self.tensortree.completely_contract_tree(to_copy=True)
        reference_tensor = reference_tree.root[1]

        self.tensortree.orthogonalize("root")
        self.tensortree.move_orthogonalization_center("root")
        self.assertEqual("root", self.tensortree.orthogonality_center_id)

        # Test for isometries
        tensor = self.tensortree.tensors["c1"]
        transfer_tensor = ptn.compute_transfer_tensor(tensor, 1)
        identity = np.eye(tensor.shape[0])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        tensor = self.tensortree.tensors["c2"]
        transfer_tensor = ptn.compute_transfer_tensor(tensor, 1)
        identity = np.eye(tensor.shape[0])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        # Test equality of tensor network
        found_tensor = self.tensortree.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(reference_tensor, found_tensor))

    def test_move_center_to_child(self):
        reference_tree = self.tensortree.completely_contract_tree(to_copy=True)
        reference_tensor = reference_tree.root[1]

        self.tensortree.orthogonalize("root")
        self.tensortree.move_orthogonalization_center("c1")
        self.assertEqual("c1", self.tensortree.orthogonality_center_id)

        # Test for isometries
        tensor = self.tensortree.tensors["root"]
        child_leg = self.tensortree.nodes["root"].neighbour_index("c2")
        transfer_tensor = ptn.compute_transfer_tensor(tensor,
                                                      (child_leg,2))
        child_leg = self.tensortree.nodes["root"].neighbour_index("c1")
        identity = np.eye(tensor.shape[child_leg])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        tensor = self.tensortree.tensors["c2"]
        transfer_tensor = ptn.compute_transfer_tensor(tensor, 1)
        identity = np.eye(tensor.shape[0])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        # Test equality of tensor network
        found_tensor = self.tensortree.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(reference_tensor, found_tensor))

    def test_move_center_to_other_child(self):
        reference_tree = self.tensortree.completely_contract_tree(to_copy=True)
        reference_tensor = reference_tree.root[1]

        self.tensortree.orthogonalize("root")
        self.tensortree.move_orthogonalization_center("c2")
        self.assertEqual("c2", self.tensortree.orthogonality_center_id)

        # Test for isometries
        tensor = self.tensortree.tensors["c1"]
        transfer_tensor = ptn.compute_transfer_tensor(tensor, 1)
        identity = np.eye(tensor.shape[0])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        tensor = self.tensortree.tensors["root"]
        child_leg = self.tensortree.nodes["root"].neighbour_index("c1")
        transfer_tensor = ptn.compute_transfer_tensor(tensor,
                                                      (child_leg,2))
        child_leg = self.tensortree.nodes["root"].neighbour_index("c2")
        identity = np.eye(tensor.shape[child_leg])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        # Test equality of tensor network
        found_tensor = self.tensortree.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(reference_tensor, found_tensor.transpose([0,2,1])))

    def test_move_center_from_child_to_parent(self):
        reference_tree = self.tensortree.completely_contract_tree(to_copy=True)
        reference_tensor = reference_tree.root[1]
        reference_tensor = np.transpose(reference_tensor,(0,2,1))

        self.tensortree.orthogonalize("c1")
        self.tensortree.move_orthogonalization_center("root")
        self.assertEqual("root", self.tensortree.orthogonality_center_id)

        # Test for isometries
        tensor = self.tensortree.tensors["c1"]
        transfer_tensor = ptn.compute_transfer_tensor(tensor, 1)
        identity = np.eye(tensor.shape[0])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        tensor = self.tensortree.tensors["c2"]
        transfer_tensor = ptn.compute_transfer_tensor(tensor, 1)
        identity = np.eye(tensor.shape[0])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        # Test equality of tensor network
        found_tensor = self.tensortree.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(reference_tensor, found_tensor))

    def test_move_center_from_child_to_other_child(self):
        reference_tree = self.tensortree.completely_contract_tree(to_copy=True)
        reference_tensor = reference_tree.root[1]

        self.tensortree.orthogonalize("c1")
        self.tensortree.move_orthogonalization_center("c2")
        self.assertEqual("c2", self.tensortree.orthogonality_center_id)

        # Test for isometries
        tensor = self.tensortree.tensors["c1"]
        transfer_tensor = ptn.compute_transfer_tensor(tensor, 1)
        identity = np.eye(tensor.shape[0])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        tensor = self.tensortree.tensors["root"]
        child_leg = self.tensortree.nodes["root"].neighbour_index("c1")
        transfer_tensor = ptn.compute_transfer_tensor(tensor,
                                                      (child_leg,2))
        child_leg = self.tensortree.nodes["root"].neighbour_index("c2")
        identity = np.eye(tensor.shape[child_leg])
        self.assertTrue(np.allclose(identity, transfer_tensor))

        # Test equality of tensor network
        found_tensor = self.tensortree.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(reference_tensor, found_tensor.transpose([0,2,1])))

class TestTreeTensorNetworkBigTree(unittest.TestCase):
    def setUp(self):
        self.ttn = ptn.TreeTensorNetwork()

        node1, tensor1 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id1")
        node2, tensor2 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id2")
        node3, tensor3 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id3")
        node4, tensor4 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id4")
        node5, tensor5 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id5")
        node6, tensor6 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id6")
        node7, tensor7 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id7")
        node8, tensor8 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id8")
        node9, tensor9 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id9")

        self.ttn.add_root(node1, tensor1)
        self.ttn.add_child_to_parent(node2, tensor2, 0, "id1", 0)
        self.ttn.add_child_to_parent(node3, tensor3, 3, "id2", 3)
        self.ttn.add_child_to_parent(node4, tensor4, 0, "id3", 1)
        self.ttn.add_child_to_parent(node5, tensor5, 2, "id2", 3)
        self.ttn.add_child_to_parent(node6, tensor6, 0, "id5", 1)
        self.ttn.add_child_to_parent(node7, tensor7, 1, "id5", 2)
        self.ttn.add_child_to_parent(node8, tensor8, 1, "id1", 1)
        self.ttn.add_child_to_parent(node9, tensor9, 2, "id8", 2)

    def test_absorb_into_open_legs(self):
        tensor_shape_dict = {"id1": (4, 5, 4, 5),
                             "id2": (3, 3),
                             "id3": (3, 4, 3, 4),
                             "id4": (3, 4, 5, 3, 4, 5),
                             "id5": (5, 5),
                             "id6": (3, 4, 5, 3, 4, 5),
                             "id7": (2, 4, 5, 2, 4, 5),
                             "id8": (2, 5, 2, 5),
                             "id9": (2, 3, 5, 2, 3, 5)}
        node_open_leg_values = {"id1": [2, 3],
                                "id2": [3],
                                "id3": [2, 3],
                                "id4": [1, 2, 3],
                                "id5": [3],
                                "id6": [1, 2, 3],
                                "id7": [1, 2, 3],
                                "id8": [2, 3],
                                "id9": [1, 2, 3]}

        for node_id, node_tensor in self.ttn.tensors.items():
            tensor = ptn.crandn(tensor_shape_dict[node_id])
            ref_tensor = deepcopy(node_tensor)
            tensor_legs = list(range(int(tensor.ndim / 2)))
            ref_tensor = np.tensordot(ref_tensor, tensor,
                                      axes=(node_open_leg_values[node_id], tensor_legs))

            self.ttn.absorb_into_open_legs(node_id, tensor)
            found_tensor = self.ttn.tensors[node_id]

            self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_tensor_contraction_leaf_only_child(self):
        self.ttn.contract_nodes("id8", "id9")
        contracted_node, contracted_tensor = self.ttn["id8contrid9"]

        # Test Node
        self.assertTrue(isinstance(contracted_node, ptn.Node))
        self.assertEqual(contracted_node.children, [])
        self.assertEqual(contracted_node.parent, "id1")

        # Test Tensor
        correct_shape = (3, 2, 5, 2, 3, 5)
        self.assertEqual(correct_shape, contracted_tensor.shape)

        # Test Connectivity
        self.assertTrue(self.ttn.is_parent_of("id1", "id8contrid9"))

    def test_tensor_contraction_leaf_not_only_child(self):
        self.ttn.contract_nodes("id5", "id7")
        contracted_id = "id5contrid7"
        contracted_node, contracted_tensor = self.ttn[contracted_id]

        # Test Node
        self.assertTrue(isinstance(contracted_node, ptn.Node))
        self.assertEqual(contracted_node.children, ["id6"])
        self.assertEqual(contracted_node.parent, "id2")

        # Test Tensor
        correct_shape = (4, 2, 5, 2, 4, 5)
        self.assertEqual(correct_shape, contracted_tensor.shape)

        # Test Connectivity
        self.assertTrue(self.ttn.is_parent_of("id2", contracted_id))
        self.assertTrue(self.ttn.is_child_of("id6", contracted_id))

    def test_tensor_contraction_middle_node(self):
        self.ttn.contract_nodes("id2", "id5")
        contracted_id = "id2contrid5"
        contracted_node, contracted_tensor = self.ttn[contracted_id]

        # Test Node
        child_ids = ["id3", "id6", "id7"]
        self.assertTrue(isinstance(contracted_node, ptn.Node))
        self.assertEqual(contracted_node.children, child_ids)
        self.assertEqual(contracted_node.parent, "id1")

        # Test Tensor
        correct_shape = (2, 5, 2, 3, 3, 5)
        self.assertEqual(correct_shape, contracted_tensor.shape)

        # Test Connectivity
        self.assertTrue(self.ttn.is_parent_of("id1", contracted_id))
        for child_id in child_ids:
            self.assertTrue(self.ttn.is_child_of(child_id, contracted_id))

    def test_tensor_contraction_root(self):
        self.ttn.contract_nodes("id1", "id2")
        contracted_id = "id1contrid2"
        contracted_node, contracted_tensor = self.ttn[contracted_id]

        # Test Node
        child_ids = ["id8", "id3", "id5"]
        self.assertTrue(isinstance(contracted_node, ptn.Node))
        self.assertEqual(contracted_node.children, child_ids)
        self.assertTrue(contracted_node.is_root())

        # Test Tensor
        correct_shape = (3, 5, 4, 4, 5, 3)
        self.assertEqual(correct_shape, contracted_tensor.shape)

        # Test Connectivity
        for child_id in child_ids:
            self.assertTrue(self.ttn.is_child_of(child_id, contracted_id))

    def test_legs_before_combination_leaf_only_child(self):
        found_legs8, found_legs9 = self.ttn.legs_before_combination("id8", "id9")
        ref_legs8 = ptn.LegSpecification("id1", [], [1,2], None)
        ref_legs9 = ptn.LegSpecification(None, None, [3,4,5], None)
        self.assertEqual(ref_legs8, found_legs8)
        self.assertEqual(ref_legs9, found_legs9)

        # And reverse
        rev_legs9, rev_legs8 = self.ttn.legs_before_combination("id9", "id8")
        ref_legs8 = ptn.LegSpecification("id1", [], [4,5], None)
        ref_legs9 = ptn.LegSpecification(None, None, [1,2,3], None)
        self.assertEqual(ref_legs8, rev_legs8)
        self.assertEqual(ref_legs9, rev_legs9)

    def test_legs_before_combination_leaf_not_only_child(self):
        found_legs5, found_legs7 = self.ttn.legs_before_combination("id5", "id7")
        ref_legs5 = ptn.LegSpecification("id2", ["id6"], [2], None)
        ref_legs7 = ptn.LegSpecification(None, None, [3,4,5], None)
        self.assertEqual(ref_legs5, found_legs5)
        self.assertEqual(ref_legs7, found_legs7)

        # And reverse
        rev_legs7, rev_legs5 = self.ttn.legs_before_combination("id7", "id5")
        ref_legs5 = ptn.LegSpecification("id2", ["id6"], [5], None)
        ref_legs7 = ptn.LegSpecification(None, None, [2,3,4], None)
        self.assertEqual(ref_legs5, rev_legs5)
        self.assertEqual(ref_legs7, rev_legs7)

    def test_legs_before_combination_middle_node(self):
        found_legs2, found_legs5 = self.ttn.legs_before_combination("id2", "id5")
        ref_legs2 = ptn.LegSpecification("id1", ["id3"], [4], None)
        ref_legs5 = ptn.LegSpecification(None, ["id6", "id7"], [5], None)
        self.assertEqual(ref_legs2, found_legs2)
        self.assertEqual(ref_legs5, found_legs5)

        # And reverse
        rev_legs5, rev_legs2 = self.ttn.legs_before_combination("id5", "id2")
        ref_legs2 = ptn.LegSpecification("id1", ["id3"], [5], None)
        ref_legs5 = ptn.LegSpecification(None, ["id6", "id7"], [4], None)
        self.assertEqual(ref_legs2, rev_legs2)
        self.assertEqual(ref_legs5, rev_legs5)

    def test_legs_before_combination_root(self):
        found_legs1, found_legs2 = self.ttn.legs_before_combination("id1", "id2")
        ref_legs1 = ptn.LegSpecification(None, ["id8"], [3,4], None)
        ref_legs2 = ptn.LegSpecification(None, ["id3", "id5"], [5], None)
        self.assertEqual(ref_legs1, found_legs1)
        self.assertEqual(ref_legs2, found_legs2)

        # And reverse
        rev_legs2, rev_legs1 = self.ttn.legs_before_combination("id2", "id1")
        ref_legs1 = ptn.LegSpecification(None, ["id8"], [4,5], None)
        ref_legs2 = ptn.LegSpecification(None, ["id3", "id5"], [3], None)
        self.assertEqual(ref_legs1, rev_legs1)
        self.assertEqual(ref_legs2, rev_legs2)

    def test_tensor_split_leaf_q1parent_vs_r3open(self):
        q_legs = {"parent_leg": "id8", "child_legs": [], "open_legs": []}
        r_legs = {"parent_leg": None, "child_legs": [], "open_legs": [1, 2, 3]}
        q_legs = ptn.LegSpecification("id8", [], [])
        r_legs = ptn.LegSpecification(None, [], [1,2,3])
        self.ttn.split_node_qr("id9", q_legs, r_legs,
                               q_identifier="q9", r_identifier="r9")

        q_node, q_tensor = self.ttn["q9"]
        r_node, r_tensor = self.ttn["r9"]

        # Test Nodes
        self.assertEqual(10, len(self.ttn.nodes))
        self.assertTrue("q9" in self.ttn.nodes)
        self.assertTrue("r9" in self.ttn.nodes)
        self.assertTrue(q_node.is_parent_of("r9"))
        self.assertTrue(r_node.is_child_of("q9"))

        # Test Tensors
        self.assertEqual((4, r_tensor.shape[0]), q_tensor.shape)
        self.assertEqual((q_tensor.shape[1], 2, 3, 5), r_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=(1, 1))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[0]), found_identity))

    def test_tensor_splitqr_leaf_r3open_vs_q1parent(self):
        r_legs = {"parent_leg": "id8", "child_legs": [], "open_legs": []}
        q_legs = {"parent_leg": None, "child_legs": [], "open_legs": [1, 2, 3]}
        r_legs = ptn.LegSpecification("id8", [], [])
        q_legs = ptn.LegSpecification(None, [], [1,2,3])
        self.ttn.split_node_qr("id9", q_legs, r_legs,
                               q_identifier="q9", r_identifier="r9")

        q_node, q_tensor = self.ttn["q9"]
        r_node, r_tensor = self.ttn["r9"]

        # Test Nodes
        self.assertEqual(10, len(self.ttn.nodes))
        self.assertTrue("q9" in self.ttn.nodes)
        self.assertTrue("r9" in self.ttn.nodes)
        self.assertTrue(q_node.is_child_of("r9"))
        self.assertTrue(r_node.is_parent_of("q9"))

        # Test Tensors
        self.assertEqual((4, q_tensor.shape[0]), r_tensor.shape)
        self.assertEqual((r_tensor.shape[1], 2, 3, 5), q_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=([1, 2, 3], [1, 2, 3]))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[0]), found_identity))

    def test_tensor_splitqr_node_q1parent1open_vs_r1child1open(self):
        q_legs = ptn.LegSpecification("id1", [], [2])
        r_legs = ptn.LegSpecification(None, ["id9"], [3])
        self.ttn.split_node_qr("id8", q_legs, r_legs,
                               q_identifier="q8", r_identifier="r8")

        q_node, q_tensor = self.ttn["q8"]
        r_node, r_tensor = self.ttn["r8"]

        # Test Nodes
        self.assertEqual(10, len(self.ttn.nodes))
        self.assertEqual(10, len(self.ttn.tensors))
        self.assertTrue("q8" in self.ttn.nodes)
        self.assertTrue("r8" in self.ttn.nodes)
        self.assertTrue(q_node.is_parent_of("r8"))
        self.assertTrue(q_node.is_child_of("id1"))
        self.assertTrue(self.ttn.nodes["id1"].is_parent_of("q8"))
        self.assertTrue(r_node.is_child_of("q8"))
        self.assertTrue(r_node.is_parent_of("id9"))
        self.assertTrue(self.ttn.nodes["id9"].is_child_of("r8"))

        # Test Tensors
        self.assertEqual((3, r_tensor.shape[0], 2), q_tensor.shape)
        self.assertEqual((q_tensor.shape[1], 4, 5), r_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=([0, 2], [0, 2]))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[1]), found_identity))

    def test_tensor_splitqr_node_q1parent1child_vs_r1child1open(self):
        q_legs = ptn.LegSpecification("id2", ["id6"], [])
        r_legs = ptn.LegSpecification(None, ["id7"], [3])

        self.ttn.split_node_qr("id5", q_legs, r_legs,
                               q_identifier="q5", r_identifier="r5")

        q_node, q_tensor = self.ttn["q5"]
        r_node, r_tensor = self.ttn["r5"]

        # Test Nodes
        self.assertEqual(10, len(self.ttn.nodes))
        self.assertTrue("q5" in self.ttn.nodes)
        self.assertTrue("r5" in self.ttn.nodes)
        self.assertTrue(q_node.is_parent_of("r5"))
        self.assertTrue(q_node.is_child_of("id2"))
        self.assertTrue(self.ttn.nodes["id2"].is_parent_of("q5"))
        self.assertTrue(q_node.is_parent_of("id6"))
        self.assertTrue(self.ttn.nodes["id6"].is_child_of("q5"))
        self.assertTrue(r_node.is_child_of("q5"))
        self.assertTrue(r_node.is_parent_of("id7"))
        self.assertTrue(self.ttn.nodes["id7"].is_child_of("r5"))

        # Test Tensors
        self.assertEqual((4, r_tensor.shape[0], 2), q_tensor.shape)
        self.assertEqual((q_tensor.shape[1], 3, 5), r_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=([0, 2], [0, 2]))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[1]), found_identity))

    def test_tensor_splitqr_root_q1child1open_vs_r1child1open(self):
        q_legs = ptn.LegSpecification(None, ["id2"], [2], None)
        r_legs = ptn.LegSpecification(None, ["id8"], [3], None)
        self.ttn.split_node_qr("id1", q_legs, r_legs,
                               q_identifier="q1", r_identifier="r1")

        q_node, q_tensor = self.ttn["q1"]
        r_node, r_tensor = self.ttn["r1"]

        # Test Nodes
        self.assertEqual(10, len(self.ttn.nodes))
        self.assertTrue("q1" in self.ttn.nodes)
        self.assertTrue("r1" in self.ttn.nodes)
        self.assertTrue(q_node.is_parent_of("r1"))
        self.assertTrue(q_node.is_root())
        self.assertTrue(self.ttn.root_id == q_node.identifier)
        self.assertTrue(q_node.is_parent_of("id2"))
        self.assertTrue(self.ttn.nodes["id2"].is_child_of("q1"))
        self.assertTrue(r_node.is_child_of("q1"))
        self.assertTrue(r_node.is_parent_of("id8"))
        self.assertTrue(self.ttn.nodes["id8"].is_child_of("r1"))

        # Test Tensors
        self.assertEqual((r_tensor.shape[0], 2, 4), q_tensor.shape)
        self.assertEqual((q_tensor.shape[0], 3, 5), r_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=([1, 2], [1, 2]))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[0]), found_identity))

# TODO: Reactivate later
# def test_apply_hamiltonian(self):
#     ttns = ptn.TreeTensorNetwork()
#     node1, tensor1 = ptn.random_tensor_node((2, 3), identifier="site1")
#     node2, tensor2 = ptn.random_tensor_node((2, 4), identifier="site2")

#     ttns.add_root(node1, tensor1)
#     ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)

#     full_state = ttns.completely_contract_tree(to_copy=True).tensors["site1_contr_site2"]

#     term1 = {"site1": "11", "site2": "21"}
#     term2 = {"site1": "12", "site2": "22"}
#     conversion_dict = {"11": ptn.crandn((3, 3)), "12": ptn.crandn((3, 3)),
#                        "21": ptn.crandn((4, 4)), "22": ptn.crandn((4, 4))}

#     hamiltonian = ptn.Hamiltonian(terms=[term1, term2],
#                                   conversion_dictionary=conversion_dict)

#     ttns.apply_hamiltonian(hamiltonian, conversion_dict)
#     test_output = ttns.completely_contract_tree(to_copy=True).tensors["site1_contr_site2"]

#     full_tensor = hamiltonian.to_tensor(ttns)
#     ttno_output = np.tensordot(full_state, full_tensor, axes=([0, 1], [2, 3]))
#     np.allclose(test_output, ttno_output)


if __name__ == "__main__":
    unittest.main()
