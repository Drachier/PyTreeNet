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

        self.assertEqual(len(self.tensortree.nodes),2)
        self.assertEqual(self.tensortree.nodes["child1"],self.node2)

        self.tensortree.add_child_to_parent(self.node3, 0, "child1", 0)

        self.assertEqual(len(self.tensortree.nodes),3)
        self.assertEqual(self.tensortree.nodes["child2"],self.node3)

    def test_parent_to_root(self):
        self.tensortree.add_root(self.node1)
        self.tensortree.add_child_to_parent(self.node2, 1, "orig_root", 1)
        self.tensortree.add_child_to_parent(self.node3, 0, "child1", 0)

        self.tensortree.add_parent_to_root(0, self.node4, 0)

        self.assertEqual(self.tensortree.root_id, "new_root")
        self.assertEqual(len(self.tensortree.nodes),4)
        self.assertEqual(self.tensortree.nodes["new_root"],self.node4)

    def test_combine_nodes(self):
        pass
        #TODO: Test

    def test_distance_to_node(self):
        node1 = ptn.random_tensor_node((2,3,4,5), identifier="id1")
        node2 = ptn.random_tensor_node((2,3,4,5), identifier="id2")
        node3 = ptn.random_tensor_node((2,3,4,5), identifier="id3")
        node4 = ptn.random_tensor_node((2,3,4,5), identifier="id4")
        node5 = ptn.random_tensor_node((2,3,4,5), identifier="id5")
        node6 = ptn.random_tensor_node((2,3,4,5), identifier="id6")
        node7 = ptn.random_tensor_node((2,3,4,5), identifier="id7")
        node8 = ptn.random_tensor_node((2,3,4,5), identifier="id8")
        node9 = ptn.random_tensor_node((2,3,4,5), identifier="id9")

        self.tensortree.add_root(node1)
        self.tensortree.add_child_to_parent(node2, 0, "id1", 0)
        self.tensortree.add_child_to_parent(node3, 3, "id2", 3)

        distance_dict = self.tensortree.distance_to_node("id2")
        self.assertAlmostEqual(min(distance_dict.values()), 0)
        self.assertAlmostEqual(max(distance_dict.values()), 1)

        self.tensortree.add_child_to_parent(node4, 0, "id3", 0)
        self.tensortree.add_child_to_parent(node5, 2, "id2", 2)
        self.tensortree.add_child_to_parent(node6, 0, "id5", 0)
        self.tensortree.add_child_to_parent(node7, 1, "id5", 1)
        self.tensortree.add_child_to_parent(node8, 1, "id1", 1)
        self.tensortree.add_child_to_parent(node9, 2, "id8", 2)

        self.assertEqual(len(self.tensortree.nodes), 9)

        distance_dict = self.tensortree.distance_to_node("id2")
        self.assertAlmostEqual(min(distance_dict.values()), 0)
        self.assertAlmostEqual(max(distance_dict.values()), 3)

        ref_distance_dict = {"id2":0,
                             "id1":1,
                             "id3":1,
                             "id5":1,
                             "id4":2,
                             "id6":2,
                             "id7":2,
                             "id8":2,
                             "id9":3}
        self.assertEqual(distance_dict, ref_distance_dict)


if __name__ == "__main__":
    unittest.main()