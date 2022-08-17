import unittest

import pytreenet as ptn
from pytreenet.contract_node import _construct_contracted_identifier

class TestContractNode(unittest.TestCase):

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

    def test_construct_contracted_identifier(self):
        new_id1 = _construct_contracted_identifier("id1", "id2")
        self.assertEqual(new_id1, "id1_contr_id2")

        new_id2 = _construct_contracted_identifier("id1", "id2", "new")
        self.assertEqual(new_id2, "new")

    def test_find_connecting_legs(self):
        leg2to4, leg4to2 = ptn.find_connecting_legs(self.node2, self.node4)
        self.assertEqual(leg2to4, 2)
        self.assertEqual(leg4to2, 0)

if __name__ == "__main__":
    unittest.main()
