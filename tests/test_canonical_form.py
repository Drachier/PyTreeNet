import unittest
import numpy as np

from math import prod

import pytreenet as ptn

from pytreenet.canonical_form import _find_smallest_distance_neighbour

class TestTreeTensorNetworkSimple(unittest.TestCase):
    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        self.node1 = ptn.random_tensor_node((2,3), identifier="node1")
        self.node2 = ptn.random_tensor_node((2,3,4), identifier="node2")

        self.tree_tensor_network.add_root(self.node1)
        self.tree_tensor_network.add_child_to_parent(self.node2, 0, "node1", 0)

    def testsimple_find_smallest_distance_neighbout(self):
        distance_dict = self.tree_tensor_network.distance_to_node("node1")
        minimum_distance_neighbour_id = _find_smallest_distance_neighbour(self.node2, distance_dict)
        self.assertEqual("node1", minimum_distance_neighbour_id)

    def testsimple_canonical_form(self):
        reference_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        ref_tensor = reference_ttn.nodes[reference_ttn.root_id].tensor

        ptn.canonical_form(self.tree_tensor_network, "node1")

        node1 = self.tree_tensor_network.nodes["node1"]
        node2 = self.tree_tensor_network.nodes["node2"]
        print(node1.tensor.shape)
        print(node2.tensor.shape)

        result_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        result_tensor = result_ttn.nodes[result_ttn.root_id].tensor

        self.assertFalse(result_ttn == reference_ttn)
        self.assertTrue(np.allclose(ref_tensor,result_tensor))

        node2 = self.tree_tensor_network.nodes["node2"]
        tensor2 = node2.tensor
        parent_leg2 = node2.parent_leg
        parent_dimension2 = tensor2.shape[parent_leg2[1]]
        identity = np.eye(parent_dimension2)

        open_indices2 = node2.open_legs
        transfer_tensor = ptn.compute_transfer_tensor(tensor2, open_indices2)

        self.assertEqual(transfer_tensor.shape, (2,2))
        self.assertTrue(np.allclose(identity, transfer_tensor))


class TestTreeTensorNetworkComplicated(unittest.TestCase):
    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        # Constructing a tree for tests
        self.node1 = ptn.random_tensor_node((2,2), identifier="node1")
        self.node2 = ptn.random_tensor_node((2,3,3), identifier="node2")
        self.node3 = ptn.random_tensor_node((3,), identifier="node3")
        self.node4 = ptn.random_tensor_node((3,4), identifier="node4")
        self.node5 = ptn.random_tensor_node((2,3,3), identifier="node5")
        self.node6 = ptn.random_tensor_node((3,4,4), identifier="node6")
        self.node7 = ptn.random_tensor_node((4,2,2), identifier="node7")

        self.tree_tensor_network.add_root(self.node1)
        self.tree_tensor_network.add_child_to_parent(self.node2, 0, "node1", 0)
        self.tree_tensor_network.add_child_to_parent(self.node3, 0, "node2", 1)
        self.tree_tensor_network.add_child_to_parent(self.node4, 0, "node2", 2)
        self.tree_tensor_network.add_child_to_parent(self.node5, 0, "node1", 1)
        self.tree_tensor_network.add_child_to_parent(self.node6, 0, "node5", 2)
        self.tree_tensor_network.add_child_to_parent(self.node7, 0, "node6", 1)

    # def test_canonical_form(self):
    #     reference_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
    #     ref_tensor = reference_ttn.nodes[reference_ttn.root_id].tensor

    #     ptn.canonical_form(self.tree_tensor_network, "node1")

    #     result_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
    #     result_tensor = result_ttn.nodes[result_ttn.root_id].tensor

    #     self.assertTrue(np.allclose(ref_tensor,result_tensor))



if __name__ == "__main__":
    unittest.main()