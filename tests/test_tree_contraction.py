import unittest

from copy import deepcopy

import numpy as np

import pytreenet as ptn

class TestTreeContraction(unittest.TestCase):

    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        # Constructing a tree for tests
        self.node1, self.tensor1 = ptn.random_tensor_node(
            (2, 3), identifier="node1")
        self.node2, self.tensor2 = ptn.random_tensor_node(
            (2, 4, 5), identifier="node2")
        self.node3, self.tensor3 = ptn.random_tensor_node(
            (4,), identifier="node3")
        self.node4, self.tensor4 = ptn.random_tensor_node(
            (5, 8), identifier="node4")
        self.node5, self.tensor5 = ptn.random_tensor_node(
            (3, 6, 7), identifier="node5")
        self.node6, self.tensor6 = ptn.random_tensor_node(
            (7, 10, 9), identifier="node6")
        self.node7, self.tensor7 = ptn.random_tensor_node(
            (10, 11, 12), identifier="node7")

        self.tree_tensor_network.add_root(self.node1, self.tensor1)
        self.tree_tensor_network.add_child_to_parent(
            self.node2, self.tensor2, 0, "node1", 0)
        self.tree_tensor_network.add_child_to_parent(
            self.node3, self.tensor3, 0, "node2", 1)
        self.tree_tensor_network.add_child_to_parent(
            self.node4, self.tensor4, 0, "node2", 2)
        self.tree_tensor_network.add_child_to_parent(
            self.node5, self.tensor5, 0, "node1", 1)
        self.tree_tensor_network.add_child_to_parent(
            self.node6, self.tensor6, 0, "node5", 2)
        self.tree_tensor_network.add_child_to_parent(
            self.node7, self.tensor7, 0, "node6", 1)

        # Generate a 2nd TTN with the same structure but different tensors
        self.tree_tensor_network2 = deepcopy(self.tree_tensor_network)

        for node_id, tensor in self.tree_tensor_network2.tensors.items():
            shape = tensor.shape
            self.tree_tensor_network2.tensors[node_id] = ptn.crandn(shape)

        # Setting up a simple TTN
        self.simple_ttn1 = ptn.TreeTensorNetwork()

        node1, tensor1 = ptn.random_tensor_node((2, 3), identifier="node1")
        node2, tensor2 = ptn.random_tensor_node((2, 4, 5), identifier="node2")
        node3, tensor3 = ptn.random_tensor_node((3, 6), identifier="node3")

        self.simple_ttn1.add_root(node1, tensor1)
        self.simple_ttn1.add_child_to_parent(node2, tensor2, 0, "node1", 0)
        self.simple_ttn1.add_child_to_parent(node3, tensor3, 0, "node1", 1)

        # Generate a 2nd TTN with the same structure but different tensors
        self.simple_ttn2 = deepcopy(self.simple_ttn1)

        for node_id, tensor in self.simple_ttn2.tensors.items():
            shape = tensor.shape
            self.simple_ttn2.tensors[node_id] = ptn.crandn(shape)

    def test_completely_contract_tree_simple(self):
        result_ttn = ptn.completely_contract_tree(
            self.simple_ttn1, to_copy=True)
        root_id = result_ttn.root_id
        result_node, result_tensor = result_ttn[root_id]

        # The contraction order should not matter for the final result
        # except for leg_order
        self.simple_ttn1.contract_nodes("node3", "node1", new_identifier="r13")
        self.simple_ttn1.contract_nodes("node2", "r13", new_identifier="root")
        ref_tensor = self.simple_ttn1.tensors["root"]

        # Test Tensor
        self.assertEqual(ref_tensor.shape, result_tensor.shape)
        self.assertTrue(np.allclose(ref_tensor, result_tensor))

        # Test Nodes
        self.assertEqual(1, len(result_ttn.nodes))
        self.assertEqual(1, len(result_ttn.tensors))
        self.assertTrue(result_node.is_root())
        self.assertTrue(result_node.is_leaf())

    def test_completely_contract_tree(self):
        result_ttn = ptn.completely_contract_tree(
            self.tree_tensor_network, to_copy=True)
        root_id = result_ttn.root_id
        result_node, result_tensor = result_ttn[root_id]

        # The contraction order should not matter for the final result
        # Except for leg_order
        self.tree_tensor_network.contract_nodes("node6", "node7", new_identifier="67")
        self.tree_tensor_network.contract_nodes("node5", "node1", new_identifier="15")
        self.tree_tensor_network.contract_nodes("node3", "node2", new_identifier="23")
        self.tree_tensor_network.contract_nodes("23", "node4", new_identifier="234")
        self.tree_tensor_network.contract_nodes("67", "15", new_identifier="root")
        self.tree_tensor_network.contract_nodes("root", "234", new_identifier="root")
        root_id = self.tree_tensor_network.root_id
        _, ref_tensor = self.tree_tensor_network[root_id]
        ref_tensor = np.transpose(ref_tensor, axes=(4,3,0,1,2))

        # Test Tensor
        self.assertTrue(ref_tensor.shape, result_tensor.shape)
        self.assertTrue(np.allclose(result_tensor, ref_tensor))

        # Test Node
        self.assertEqual(1, len(result_ttn.nodes))
        self.assertEqual(1, len(result_ttn.tensors))
        self.assertTrue(result_node.is_root())
        self.assertTrue(result_node.is_leaf())

    def test_contract_two_ttn_simple(self):
        # We get the correct result via manual contraction
        result1 = ptn.completely_contract_tree(self.simple_ttn1)
        result2 = ptn.completely_contract_tree(self.simple_ttn2)
        tensor1 = result1[result1.root_id][1]
        tensor2 = result2[result2.root_id][1]

        correct_result = np.tensordot(tensor1, tensor2, axes=([0,1,2], [0,1,2]))

        found_result = ptn.contract_two_ttn(self.simple_ttn1, self.simple_ttn2)

        self.assertTrue(np.allclose(correct_result, found_result))

    def test_contract_two_ttn_complicated(self):   
        work_ttn1 = deepcopy(self.tree_tensor_network)
        work_ttn2 = deepcopy(self.tree_tensor_network2)

        found_result = ptn.contract_two_ttn(work_ttn1, work_ttn2)

        # Since both have the same structure, we can completely contract them and take the scalar product
        ttn1_contr = ptn.completely_contract_tree(self.tree_tensor_network,
                                                  to_copy=True)
        ttn1_tensor = ttn1_contr.tensors[ttn1_contr.root_id]
        ttn2_contr = ptn.completely_contract_tree(self.tree_tensor_network2,
                                                  to_copy=True)
        ttn2_tensor = ttn2_contr.tensors[ttn2_contr.root_id]
        all_axes = range(ttn1_tensor.ndim)
        correct_result = np.tensordot(ttn1_tensor, ttn2_tensor,
                                      axes=(all_axes, all_axes))
        self.assertTrue(np.allclose(correct_result, found_result))

if __name__ == "__main__":
    unittest.main()
