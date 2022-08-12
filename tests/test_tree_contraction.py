import unittest
import pytreenet as ptn

class TestTreeTensorNetworkBasics(unittest.TestCase):

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

    def test_combine_nodes(self):
        print(self.node2.tag)
        self.assertRaises(ptn.NoConnectionException ,
                          ptn.combine_nodes(self.tree_tensor_network, "node1", "node3"))

if __name__ == "__main__":
    unittest.main()