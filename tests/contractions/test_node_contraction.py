"""
Test the node contraction functionality in PyTreeNet.
"""
import unittest
from copy import deepcopy

import numpy as np

from pytreenet.contractions.node_contraction import contract_nodes
from pytreenet.random.random_node import random_tensor_node

class TestNodeContraction(unittest.TestCase):
    """
    Test the node contraction functionality in PyTreeNet.
    """

    def test_contract_vectors(self):
        """
        Test the contraction of two vector-like nodes.
        """
        node1, tensor1 = random_tensor_node((2, ), identifier="node1")
        node2, tensor2 = random_tensor_node((2, ), identifier="node2")
        node1.open_leg_to_child(node2.identifier, 0)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                              node2, tensor2,
                                              new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_root())
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, ())
        self.assertEqual(new_tensor.shape, ())
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(0,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_vec_mat(self):
        """
        Test the contraction of a vector-like node connected to a matrix-like node,
        where the other leg is open.
        """
        node1, tensor1 = random_tensor_node((2, ), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 3), identifier="node2")
        node1.open_leg_to_child(node2.identifier, 0)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                              node2, tensor2,
                                              new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_root())
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, (3,))
        self.assertEqual(new_tensor.shape, (3,))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(0,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_mat_vec(self):
        """
        Test the contraction of a matrix-like node connected to a vector-like node,
        where the other leg is open. Basically the parentage is reversed compared
        to the previous test.
        """
        node1, tensor1 = random_tensor_node((2, 3), identifier="node1")
        node2, tensor2 = random_tensor_node((2, ), identifier="node2")
        node1.open_leg_to_child(node2.identifier, 0)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                              node2, tensor2,
                                              new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_root())
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, (3,))
        self.assertEqual(new_tensor.shape, (3,))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(0,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_contract_matrices(self):
        """
        Test the contraction of two matrix-like nodes.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((3, 4), identifier="node2")
        node1.open_leg_to_child(node2.identifier, 0)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                              node2, tensor2,
                                              new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_root())
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, (2, 4))
        self.assertEqual(new_tensor.shape, (2, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(0,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_contract_matrices_reverse(self):
        """
        Test the contraction of two matrix-like nodes, but with the parentage reversed.
        This is the same as the previous test, but the first node is the child and the
        second node is the parent as given to the function.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((3, 4), identifier="node2")
        node1.open_leg_to_child(node2.identifier, 0)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                              node1, tensor1,
                                              new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_root())
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, (4, 2))
        # To have the tensor in the correct shape, it needs to be transposed via the node.
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (4, 2))
        # Reference contraction
        ref = np.tensordot(ref_tens[1], ref_tens[0], axes=(0,0))
        np.testing.assert_array_almost_equal(new_tensor, ref)

    def test_parent_has_parent_no_open(self):
        """
        Test the contraction in the case where the parent node has a parent 
        node and neither node has an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, ), identifier="node2")
        parent_id = "parent"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                              node2, tensor2,
                                              new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        print(new_node.children)
        self.assertTrue(new_node.is_leaf())
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertEqual(new_node.shape, (3,))
        self.assertEqual(new_tensor.shape, (3,))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_parent_popen(self):
        """
        Test the contraction in the case where the parent node has a parent
        node and an open leg, while the child node has no open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 4), identifier="node1")
        node2, tensor2 = random_tensor_node((2, ), identifier="node2")
        parent_id = "parent"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                            node2, tensor2,
                                            new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, (3, 4))
        self.assertEqual(new_tensor.shape, (3, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_parent_copen(self):
        """
        Test the contraction in the case where the parent node has a parent
        node and no open leg, while the child node has an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        parent_id = "parent"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                            node2, tensor2,
                                            new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, (3, 4))
        self.assertEqual(new_tensor.shape, (3, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_parent_copen_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        node and no open leg, while the child node has an open leg. This is
        the same as the previous test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        parent_id = "parent"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                            node2, tensor2,
                                            new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, (3, 4))
        # To have the tensor in the correct shape, it needs to be transposed via the node.
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_parent_bothopen(self):
        """
        Test the contraction in the case where the parent node has a parent
        node and both nodes have an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        parent_id = "parent"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                            node2, tensor2,
                                            new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, (3, 5, 4))
        # To have the tensor in the correct shape, it needs to be transposed via the node.
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 5, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_parent_bothopen_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        node and both nodes have an open leg. This is the same as the previous
        test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        parent_id = "parent"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                              node1, tensor1,
                                              new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_leaf())
        self.assertEqual(new_node.shape, (3, 4, 5))
        # To have the tensor in the correct shape, it needs to be transposed via the node.
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        ## The open leg must be in the last position, but argument order
        ref = ref.transpose(0, 2, 1)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_child_no_open(self):
        """
        Test the contraction in the case where the parent node has a child
        node and neither node has an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, ), identifier="node2")
        child_id = "childp1"
        node1.open_leg_to_child(child_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_root())
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3,))
        self.assertEqual(new_tensor.shape, (3,))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)


    def test_parent_has_child_popen(self):
        """
        Test the contraction in the case where the parent node has a child
        node and an open leg, while the child node has no open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 4), identifier="node1")
        node2, tensor2 = random_tensor_node((2, ), identifier="node2")
        child_id = "childp1"
        node1.open_leg_to_child(child_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (3, 4))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_child_copen(self):
        """
        Test the contraction in the case where the parent node has a child
        node and no open leg, while the child node has an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        child_id = "childp1"
        node1.open_leg_to_child(child_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (3, 4))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_child_copen_rev(self):
        """
        Test the contraction in the case where the parent node has a child
        node and no open leg, while the child node has an open leg. This is
        the same as the previous test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        child_id = "childp1"
        node1.open_leg_to_child(child_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (3, 4))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_child_bothopen(self):
        """
        Test the contraction in the case where the parent node has a child
        node and both nodes have an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        child_id = "childp1"
        node1.open_leg_to_child(child_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (3, 5, 4))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 5, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_child_bothopen_rev(self):
        """
        Test the contraction in the case where the parent node has a child
        node and both nodes have an open leg. This is the same as the previous
        test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        child_id = "childp1"
        node1.open_leg_to_child(child_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        ## The open leg must be in the last position, but argument order
        ref = ref.transpose(0, 2, 1)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandc_no_open(self):
        """
        Test the contraction in the case where the parent node has a parent
        and a child node, and neither node has an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 4), identifier="node1")
        node2, tensor2 = random_tensor_node((2,), identifier="node2")
        parent_id = "parent"
        child_id = "childp1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node1.open_leg_to_child(child_id, 2)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4,))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4,))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandc_popen(self):
        """
        Test the contraction in the case where the parent node has a parent
        and a child node, and the parent node has an open leg, while the child
        node has no open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 4, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, ), identifier="node2")
        parent_id = "parent"
        child_id = "childp1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node1.open_leg_to_child(child_id, 2)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandc_copen(self):
        """
        Test the contraction in the case where the parent node has a parent
        and a child node, and the parent node has no open leg, while the child
        node has an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 4), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 5), identifier="node2")
        parent_id = "parent"
        child_id = "childp1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node1.open_leg_to_child(child_id, 2)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandc_copen_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        and a child node, and the parent node has no open leg, while the child
        node has an open leg. This is the same as the previous test, but with
        reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2, 4), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 5), identifier="node2")
        parent_id = "parent"
        child_id = "childp1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node1.open_leg_to_child(child_id, 2)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandc_bothopen(self):
        """
        Test the contraction in the case where the parent node has a parent
        and a child node, and both nodes have an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 4, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 6), identifier="node2")
        parent_id = "parent"
        child_id = "childp1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node1.open_leg_to_child(child_id, 2)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 5, 6))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5, 6))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandc_bothopen_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        and a child node, and both nodes have an open leg. This is the same as
        the previous test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2, 4, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 6), identifier="node2")
        parent_id = "parent"
        child_id = "childp1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node1.open_leg_to_child(child_id, 2)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 6, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 6, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        ## The open leg must be in the last position, but argument order
        ref = ref.transpose(0, 1, 3, 2)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandcc_no_open(self):
        """
        Test the contraction in the case where the parent node has a parent
        and two child nodes, and neither node has an open leg.
        """
        node1, tensor1 = random_tensor_node(((3,4,5,2)), identifier="node1")
        node2, tensor2 = random_tensor_node((2,), identifier="node2")
        parent_id = "parent"
        child_id1 = "childp1"
        child_id2 = "childp2"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(child_id1, 1)
        node1.open_leg_to_child(child_id2, 2)
        node1.open_leg_to_child(node2.identifier, 3)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of([child_id1, child_id2]))
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(3,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandcc_popen(self):
        """
        Test the contraction in the case where the parent node has a parent
        and two child nodes, and the parent node has an open leg, while the
        child nodes have no open leg.
        """
        node1, tensor1 = random_tensor_node((3, 4, 5, 2, 6), identifier="node1")
        node2, tensor2 = random_tensor_node((2,), identifier="node2")
        parent_id = "parent"
        child_id1 = "childp1"
        child_id2 = "childp2"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(child_id1, 1)
        node1.open_leg_to_child(child_id2, 2)
        node1.open_leg_to_child(node2.identifier, 3)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of([child_id1, child_id2]))
        self.assertEqual(new_node.shape, (3, 4, 5, 6))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5, 6))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(3,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandcc_copen(self):
        """
        Test the contraction in the case where the parent node has a parent
        and two child nodes, and the parent node has no open leg, while the
        child node has an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 4, 5, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 6), identifier="node2")
        parent_id = "parent"
        child_id1 = "childp1"
        child_id2 = "childp2"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(child_id1, 1)
        node1.open_leg_to_child(child_id2, 2)
        node1.open_leg_to_child(node2.identifier, 3)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of([child_id1, child_id2]))
        self.assertEqual(new_node.shape, (3, 4, 5, 6))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5, 6))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(3,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandcc_copen_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        and two child nodes, and the parent node has no open leg, while the
        child node has an open leg. This is the same as the previous test,
        but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 4, 5, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 6), identifier="node2")
        parent_id = "parent"
        child_id1 = "childp1"
        child_id2 = "childp2"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(child_id1, 1)
        node1.open_leg_to_child(child_id2, 2)
        node1.open_leg_to_child(node2.identifier, 3)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of([child_id1, child_id2]))
        self.assertEqual(new_node.shape, (3, 4, 5, 6))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5, 6))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(3,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandcc_bothopen(self):
        """
        Test the contraction in the case where the parent node has a parent
        and two child nodes, and both nodes have an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 4, 5, 2, 6), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 7), identifier="node2")
        parent_id = "parent"
        child_id1 = "childp1"
        child_id2 = "childp2"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(child_id1, 1)
        node1.open_leg_to_child(child_id2, 2)
        node1.open_leg_to_child(node2.identifier, 3)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of([child_id1, child_id2]))
        self.assertEqual(new_node.shape, (3, 4, 5, 6, 7))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5, 6, 7))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(3,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_pandcc_bothopen_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        and two child nodes, and both nodes have an open leg. This is the same
        as the previous test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 4, 5, 2, 6), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 7), identifier="node2")
        parent_id = "parent"
        child_id1 = "childp1"
        child_id2 = "childp2"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(child_id1, 1)
        node1.open_leg_to_child(child_id2, 2)
        node1.open_leg_to_child(node2.identifier, 3)
        node2.open_leg_to_parent(node1.identifier, 0)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of([child_id1, child_id2]))
        self.assertEqual(new_node.shape, (3, 4, 5, 7, 6))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5, 7, 6))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(3,0))
        ## The open leg must be in the last position, but argument order
        ref = ref.transpose(0, 1, 2, 4, 3)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_child_has_c_no_open(self):
        """
        Test the contraction in the case where the parent node has no other
        neighbours and the child node has a child itself. Neither node has an
        open leg.
        """
        node1, tensor1 = random_tensor_node((2, ), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        child_id = "childc1"
        node1.open_leg_to_child(node2.identifier, 0)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (4, ))
        self.assertEqual(new_tensor.shape, (4, ))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(0,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_child_has_c_popen(self):
        """
        Test the contraction in the case where the parent node has no other
        neighbours and the child node has a child itself. The parent node has
        an open leg, while the child node has no open leg.
        """
        node1, tensor1 = random_tensor_node((2, 3), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        child_id = "childc1"
        node1.open_leg_to_child(node2.identifier, 0)
        node1.open_leg_to_child(child_id, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (4, 3))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (4, 3))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(0,0))
        ## A child leg will come before the open leg.
        ref = ref.transpose(1, 0)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_child_has_c_popen_rev(self):
        """
        Test the contraction in the case where the parent node has no other
        neighbours and the child node has a child itself. The parent node has
        an open leg, while the child node has no open leg. This is the same as
        the previous test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((2, 3), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        child_id = "childc1"
        node1.open_leg_to_child(node2.identifier, 0)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (4, 3))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (4, 3))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(0,0))
        ## A child leg will come before the open leg.
        ref = ref.transpose(1, 0)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_child_has_c_bothopen(self):
        """
        Test the contraction in the case where the parent node has no other
        neighbours and the child node has a child itself. Both nodes have an
        open leg.
        """
        node1, tensor1 = random_tensor_node((2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 3, 4), identifier="node2")
        child_id = "childc1"
        node1.open_leg_to_child(node2.identifier, 0)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (3, 5, 4))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 5, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(0,0))
        ## A child leg will come before the open leg.
        ## And the open legs must be in the right order.
        ref = ref.transpose(1, 0, 2)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_child_has_c_bothopen_rev(self):
        """
        Test the contraction in the case where the parent node has no other
        neighbours and the child node has a child itself. Both nodes have an
        open leg. This is the same as the previous test, but with reversed
        parentage.
        """
        node1, tensor1 = random_tensor_node((2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 3, 4), identifier="node2")
        child_id = "childc1"
        node1.open_leg_to_child(node2.identifier, 0)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertTrue(new_node.is_root())
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(0,0))
        ## A child leg will come before the open leg.
        ## And the open legs must be in the right order.
        ref = ref.transpose(1, 2, 0)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_p_child_has_c_no_open(self):
        """
        Test the contraction in the case where the parent node has a parent
        and the child node has a child itself. Neither node has an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        parent_id = "parent"
        child_id = "childc1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_p_child_has_no_open_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        and the child node has a child itself. Neither node has an open leg.
        This is the same as the previous test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        parent_id = "parent"
        child_id = "childc1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)
        
    def test_parent_has_p_child_has_c_popen(self):
        """
        Test the contraction in the case where the parent node has a parent
        and the child node has a child itself. The parent node has an open leg,
        while the child node has no open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        parent_id = "parent"
        child_id = "childc1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        ## Note that the open leg must be in the last position.
        ref = ref.transpose(0, 2, 1)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_p_child_has_c_popen_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        and the child node has a child itself. The parent node has an open leg,
        while the child node has no open leg. This is the same as the previous
        test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4), identifier="node2")
        parent_id = "parent"
        child_id = "childc1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        ## Note that the open leg must be in the last position.
        ref = ref.transpose(0, 2, 1)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_p_child_has_c_copen(self):
        """
        Test the contraction in the case where the parent node has a parent
        and the child node has a child itself. The parent node has no open leg,
        while the child node has an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4, 5), identifier="node2")
        parent_id = "parent"
        child_id = "childc1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_p_child_has_c_copen_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        and the child node has a child itself. The parent node has no open leg,
        while the child node has an open leg. This is the same as the previous
        test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4, 5), identifier="node2")
        parent_id = "parent"
        child_id = "childc1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        np.testing.assert_array_equal(new_tensor, ref)
    
    def test_parent_has_p_child_has_c_bothopen(self):
        """
        Test the contraction in the case where the parent node has a parent
        and the child node has a child itself. Both nodes have an open leg.
        """
        node1, tensor1 = random_tensor_node((3, 2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4, 6), identifier="node2")
        parent_id = "parent"
        child_id = "childc1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                                node2, tensor2,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 5, 6))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 5, 6))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        ## The child leg must come before the open leg.
        ref = ref.transpose(0, 2, 1, 3)
        np.testing.assert_array_equal(new_tensor, ref)

    def test_parent_has_p_child_has_c_bothopen_rev(self):
        """
        Test the contraction in the case where the parent node has a parent
        and the child node has a child itself. Both nodes have an open leg.
        This is the same as the previous test, but with reversed parentage.
        """
        node1, tensor1 = random_tensor_node((3, 2, 5), identifier="node1")
        node2, tensor2 = random_tensor_node((2, 4, 6), identifier="node2")
        parent_id = "parent"
        child_id = "childc1"
        node1.open_leg_to_parent(parent_id, 0)
        node1.open_leg_to_child(node2.identifier, 1)
        node2.open_leg_to_parent(node1.identifier, 0)
        node2.open_leg_to_child(child_id, 1)
        ref_tens = (deepcopy(tensor1), deepcopy(tensor2))
        new_node, new_tensor = contract_nodes(node2, tensor2,
                                                node1, tensor1,
                                                new_identifier="new_node")
        self.assertEqual(new_node.identifier, "new_node")
        self.assertTrue(new_node.is_child_of(parent_id))
        self.assertTrue(new_node.is_parent_of(child_id))
        self.assertEqual(new_node.shape, (3, 4, 6, 5))
        new_tensor = new_node.transpose_tensor(new_tensor)
        self.assertEqual(new_tensor.shape, (3, 4, 6, 5))
        # Reference contraction
        ref = np.tensordot(ref_tens[0], ref_tens[1], axes=(1,0))
        ## The child leg must come before the open leg.
        ## And the open legs must be in argument order.
        ref = ref.transpose(0, 2, 3, 1)
        np.testing.assert_array_equal(new_tensor, ref)

if __name__ == "__main__":
    unittest.main()
