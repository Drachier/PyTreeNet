# TODO: Potentially delete this test

import unittest

import numpy as np

from random import randint

import pytreenet as ptn
from pytreenet.node_contraction import (_construct_contracted_identifier,
                                        _find_connecting_legs_parent_child,
                                        _create_leg_dict
                                        )


class TestContractNode(unittest.TestCase):

    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        # Constructing a tree for tests
        self.tensornode1 = ptn.random_tensor_node((2, 2), identifier="node1")
        self.tensornode2 = ptn.random_tensor_node(
            (2, 3, 3), identifier="node2")
        self.tensornode3 = ptn.random_tensor_node((3,), identifier="node3")
        self.tensornode4 = ptn.random_tensor_node((3, 4), identifier="node4")
        self.tensornode5 = ptn.random_tensor_node(
            (2, 3, 3), identifier="node5")
        self.tensornode6 = ptn.random_tensor_node(
            (3, 4, 4), identifier="node6")
        self.tensornode7 = ptn.random_tensor_node(
            (4, 2, 2), identifier="node7")

        self.tree_tensor_network.add_root(
            self.tensornode1[0], self.tensornode1[1])
        self.tree_tensor_network.add_child_to_parent(self.tensornode2[0], self.tensornode2[1],
                                                     0, "node1", 0)
        self.tree_tensor_network.add_child_to_parent(self.tensornode3[0], self.tensornode3[1],
                                                     0, "node2", 1)
        self.tree_tensor_network.add_child_to_parent(self.tensornode4[0], self.tensornode4[1],
                                                     0, "node2", 2)
        self.tree_tensor_network.add_child_to_parent(self.tensornode5[0], self.tensornode5[1],
                                                     0, "node1", 1)
        self.tree_tensor_network.add_child_to_parent(self.tensornode6[0], self.tensornode6[1],
                                                     0, "node5", 2)
        self.tree_tensor_network.add_child_to_parent(self.tensornode7[0], self.tensornode7[1],
                                                     0, "node6", 1)

    def test_construct_contracted_identifier(self):
        new_id1 = _construct_contracted_identifier("id1", "id2")
        self.assertEqual(new_id1, "id1_contr_id2")

        new_id2 = _construct_contracted_identifier("id1", "id2", "new")
        self.assertEqual(new_id2, "new")

    def test_find_connecting_legs_parent_child(self):
        node2, _ = self.tree_tensor_network["node2"]
        node4, _ = self.tree_tensor_network["node4"]
        leg2to4, leg4to2 = _find_connecting_legs_parent_child(node2, node4)
        self.assertEqual(leg2to4, 2)
        self.assertEqual(leg4to2, 0)


class TestNodeContraction_Complicated(unittest.TestCase):

    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        # Construct tree for test
        self.node1, self.tensor1 = ptn.random_tensor_node((2, 2), identifier="node1")
        self.node2, self.tensor2 = ptn.random_tensor_node((2, 3, 3), identifier="node2")
        self.node3, self.tensor3 = ptn.random_tensor_node((3,), identifier="node3")
        self.node4, self.tensor4 = ptn.random_tensor_node((3, 4), identifier="node4")
        self.node5, self.tensor5 = ptn.random_tensor_node((2, 3, 3, 4), identifier="node5")
        self.node6, self.tensor6 = ptn.random_tensor_node((3, 4, 4), identifier="node6")
        self.node7, self.tensor7 = ptn.random_tensor_node((4, 2, 2), identifier="node7")
        self.node8, self.tensor8 = ptn.random_tensor_node((3, 4), identifier="node8")

        self.tree_tensor_network.add_root(self.node1, self.tensor1)
        self.tree_tensor_network.add_child_to_parent(self.node2, self.tensor2, 0, "node1", 0)
        self.tree_tensor_network.add_child_to_parent(self.node3, self.tensor3, 0, "node2", 1)
        self.tree_tensor_network.add_child_to_parent(self.node4, self.tensor4, 0, "node2", 2)
        self.tree_tensor_network.add_child_to_parent(self.node5, self.tensor5, 0, "node1", 1)
        self.tree_tensor_network.add_child_to_parent(self.node6, self.tensor6, 0, "node5", 2)
        self.tree_tensor_network.add_child_to_parent(self.node7, self.tensor7, 0, "node6", 1)
        self.tree_tensor_network.add_child_to_parent(self.node8, self.tensor8, 1, "node5", 3)

    def test_operator_expectation_value_on_node(self):
        shape = (2, 3, 4, 5)
        tensor = ptn.crandn(shape)

        node = ptn.TensorNode(tensor, identifier="node")

        leg_index_with_operator = 1
        matrix = ptn.crandn(
            (shape[leg_index_with_operator], shape[leg_index_with_operator]))
        operator = matrix + matrix.T

        leg_index_list = [i for i in range(
            tensor.ndim) if i != leg_index_with_operator]
        identifier_list = [str(i) for i in range(
            tensor.ndim) if i != leg_index_with_operator]
        node.open_legs_to_children(leg_index_list, identifier_list)

        found_exp_value = ptn.operator_expectation_value_on_node(
            node, operator)

        conj_tensor = tensor.conjugate()
        tensor_operator = np.tensordot(
            tensor, operator, axes=(leg_index_with_operator, 1))

        correct_exp_value = np.tensordot(tensor_operator, conj_tensor,
                                         axes=([0, 1, 2, 3], [0, 2, 3, 1]))

        self.assertAlmostEqual(correct_exp_value, found_exp_value)


if __name__ == "__main__":
    unittest.main()
