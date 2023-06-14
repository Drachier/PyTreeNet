import unittest
import pytreenet as ptn
import numpy as np


class TestTreeTensorNetworkBasics(unittest.TestCase):

    def setUp(self):
        self.tensortree = ptn.TreeTensorNetwork()
        self.node1, self.tensor1 = ptn.random_tensor_node((2, 3, 4, 5), identifier="orig_root")
        self.node2, self.tensor2 = ptn.random_tensor_node((2, 3), identifier="child1")
        self.node3, self.tensor3 = ptn.random_tensor_node((2, 3, 4, 5), identifier="child2")
        self.node4, self.tensor4 = ptn.random_tensor_node((2, 3, 4, 5), identifier="new_root")

    def test_add_root(self):
        self.assertEqual(self.tensortree.root_id, None)
        self.assertEqual(self.tensortree.nodes, dict())

        self.tensortree.add_root(self.node1, self.tensor1)

        self.assertEqual(self.tensortree.root_id, "orig_root")
        # self.assertEqual(len(self.tensortree.nodes), 1)

    def test_add_child_to_parent(self):
        self.tensortree.add_root(self.node1, self.tensor1)
        self.tensortree.add_child_to_parent(self.node2, self.tensor2, 1, "orig_root", 1)

        self.assertEqual(len(self.tensortree.nodes), 2)
        self.assertTrue(np.array_equal(self.tensortree.tensors["child1"], self.tensor2))
        self.assertEqual(self.tensortree.nodes["child1"], self.node2)

        self.tensortree.add_child_to_parent(self.node3, self.tensor3, 0, "child1", 0)

        self.assertEqual(len(self.tensortree.nodes), 3)
        self.assertTrue(np.array_equal(self.tensortree.tensors["child2"], self.tensor3))
        self.assertEqual(self.tensortree.nodes["child2"], self.node3)

    def test_parent_to_root(self):
        self.tensortree.add_root(self.node1, self.tensor1)
        self.tensortree.add_child_to_parent(self.node2, self.tensor2, 1, "orig_root", 1)
        self.tensortree.add_child_to_parent(self.node3, self.tensor3, 0, "child1", 0)

        self.tensortree.add_parent_to_root(0, self.node4, self.tensor4, 0)

        self.assertEqual(self.tensortree.root_id, "new_root")
        self.assertEqual(len(self.tensortree.nodes), 4)
        self.assertTrue(np.array_equal(self.tensortree.tensors["new_root"], self.tensor4))
        self.assertEqual(self.tensortree.nodes["new_root"], self.node4)


class TestTreeTensorNetworkBigTree(unittest.TestCase):
    def setUp(self):
        self.tensortree = ptn.TreeTensorNetwork()

        node1, tensor1 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id1")
        node2, tensor2 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id2")
        node3, tensor3 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id3")
        node4, tensor4 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id4")
        node5, tensor5 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id5")
        node6, tensor6 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id6")
        node7, tensor7 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id7")
        node8, tensor8 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id8")
        node9, tensor9 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id9")

        self.tensortree.add_root(node1, tensor1)
        self.tensortree.add_child_to_parent(node2, tensor2, 0, "id1", 0)
        self.tensortree.add_child_to_parent(node3, tensor3, 3, "id2", 3)
        self.tensortree.add_child_to_parent(node4, tensor4, 0, "id3", 0)
        self.tensortree.add_child_to_parent(node5, tensor5, 2, "id2", 2)
        self.tensortree.add_child_to_parent(node6, tensor6, 0, "id5", 0)
        self.tensortree.add_child_to_parent(node7, tensor7, 1, "id5", 1)
        self.tensortree.add_child_to_parent(node8, tensor8, 1, "id1", 1)
        self.tensortree.add_child_to_parent(node9, tensor9, 2, "id8", 2)

        self.test_node, self.test_tensor = ptn.random_tensor_node((2, 3, 4, 5), identifier="test")

    def test_distance_to_node(self):
        self.assertEqual(len(self.tensortree.nodes), 9)

        distance_dict = self.tensortree.distance_to_node("id2")
        self.assertAlmostEqual(min(distance_dict.values()), 0)
        self.assertAlmostEqual(max(distance_dict.values()), 3)

        ref_distance_dict = {"id2": 0,
                             "id1": 1,
                             "id3": 1,
                             "id5": 1,
                             "id4": 2,
                             "id6": 2,
                             "id7": 2,
                             "id8": 2,
                             "id9": 3}
        self.assertEqual(distance_dict, ref_distance_dict)

    def test_rewire_only_child(self):
        node5 = self.tensortree.nodes["id5"]

        self.tensortree.rewire_only_child("id2", "id5", "test")

        self.assertEqual(node5.parent_leg[0], "test")

    def test_rewire_only_parent(self):
        node2 = self.tensortree["id2"]
        leg_2_to_5 = node2.children_legs["id5"]

        self.tensortree.rewire_only_parent("id5", "test")

        self.assertTrue("test" in node2.children_legs)
        self.assertEqual(leg_2_to_5, node2.children_legs["test"])

    def test_find_subtree_of_node(self):
        found_subtree = self.tensortree.find_subtree_of_node("id1")
        correct_subtree = list(self.tensortree.nodes.keys())

        self.assertEqual(correct_subtree, found_subtree)

        found_subtree = self.tensortree.find_subtree_of_node("id9")
        correct_subtree = ["id9"]

        self.assertEqual(correct_subtree, found_subtree)

        found_subtree = self.tensortree.find_subtree_of_node("id2")
        correct_subtree = ["id2", "id3", "id4", "id5", "id6", "id7"]

        self.assertEqual(correct_subtree, found_subtree)

        found_subtree = self.tensortree.find_subtree_of_node("id8")
        correct_subtree = ["id8", "id9"]

        self.assertEqual(correct_subtree, found_subtree)

    def test_apply_hamiltonian(self):
        ttns = ptn.TreeTensorNetwork()
        node1, tensor1 = ptn.random_tensor_node((2, 3), identifier="site1")
        node2, tensor2 = ptn.random_tensor_node((2, 4), identifier="site2")

        ttns.add_root(node1, tensor1)
        ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)

        full_state = ttns.completely_contract_tree(to_copy=True).tensors["site1_contr_site2"]

        term1 = {"site1": "11", "site2": "21"}
        term2 = {"site1": "12", "site2": "22"}
        conversion_dict = {"11": ptn.crandn((3, 3)), "12": ptn.crandn((3, 3)),
                           "21": ptn.crandn((4, 4)), "22": ptn.crandn((4, 4))}

        hamiltonian = ptn.Hamiltonian(terms=[term1, term2],
                                      conversion_dictionary=conversion_dict)

        ttns.apply_hamiltonian(hamiltonian, conversion_dict)
        test_output = ttns.completely_contract_tree(to_copy=True).tensors["site1_contr_site2"]

        full_tensor = hamiltonian.to_tensor(ttns)
        ttno_output = np.tensordot(full_state, full_tensor, axes=([0, 1], [2, 3]))
        np.allclose(test_output, ttno_output)


if __name__ == "__main__":
    unittest.main()

node1, tensor1 = ptn.random_tensor_node((2, 3), identifier="site1")
