import unittest
import pytreenet as ptn

class TestTreeTensorNetwork(unittest.TestCase):

    def setUp(self):
        self.tensortree = ptn.TreeTensorNetwork()
        self.node1 = ptn.random_tensor_node((2,3,4,5), identifier="orig_root")
        self.node2 = ptn.random_tensor_node((2,3), identifier="child1")
        self.node3 = ptn.random_tensor_node((2,3,4), identifier="child2")
        self.node4 = ptn.random_tensor_node((2), identifier="new_root")

    def test_add_root(self):
        self.assertEqual(self.tensortree.root_id, None)
        self.assertEqual(self.tensortree.nodes, dict())

        self.tensortree.add_root(self.node1)

        self.assertEqual(self.tensortree.root_id, "orig_root")
        self.assertEqual(len(self.tensortree.nodes),1)


    def test_add_child_to_parent(self):
        self.tensortree.add_root(self.node1)
        self.tensortree.add_child_to_parent(self.node2, 1, "orig_root", 1)


if __name__ == "__main__":
    unittest.main()