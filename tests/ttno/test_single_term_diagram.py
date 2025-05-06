import unittest

import pytreenet as ptn
from pytreenet.ttno import SingleTermDiagram
from pytreenet.random import random_tensor_node

class TestStateDiagram(unittest.TestCase):

    def setUp(self):
        self.ref_tree = ptn.TreeTensorNetwork()

        node1, tensor1 = random_tensor_node((2, 2, 2), identifier="site1")
        node2, tensor2 = random_tensor_node((2, 2, 2, 2), identifier="site2")
        node5, tensor5 = random_tensor_node((2, 2, 2, 2), identifier="site5")
        node3, tensor3 = random_tensor_node((2, 2), identifier="site3")
        node4, tensor4 = random_tensor_node((2, 2), identifier="site4")
        node6, tensor6 = random_tensor_node((2, 2), identifier="site6")
        node7, tensor7 = random_tensor_node((2, 2), identifier="site7")

        self.ref_tree.add_root(node1, tensor1)
        self.ref_tree.add_child_to_parent(node2, tensor2, 0, "site1", 0)
        self.ref_tree.add_child_to_parent(node5, tensor5, 0, "site1", 1)
        self.ref_tree.add_child_to_parent(node3, tensor3, 0, "site2", 1)
        self.ref_tree.add_child_to_parent(node4, tensor4, 0, "site2", 2)
        self.ref_tree.add_child_to_parent(node6, tensor6, 0, "site5", 1)
        self.ref_tree.add_child_to_parent(node7, tensor7, 0, "site5", 2)

        self.term = {"site1": "1", "site2": "2", "site3": "3",
                     "site4": "4", "site5": "5", "site6": "6", "site7": "7"}

        self.state_diagram_empty = SingleTermDiagram(ptn.TreeTensorNetwork())

    def test_from_single_term(self):
        state_diagram = SingleTermDiagram.from_single_term(
            self.term, self.ref_tree)

        self.assertEqual(6, len(state_diagram.vertices))
        self.assertEqual(7, len(state_diagram.hyperedges))

        # Make sure every hyperedge has the correct number of vertices
        num_vertices_dict = {"site1": 2, "site2": 3, "site3": 1,
                             "site4": 1, "site5": 3, "site6": 1, "site7": 1}
        for node_id, num_vertices in num_vertices_dict.items():
            found_num_vertices = len(state_diagram.hyperedges[node_id].vertices)
            self.assertEqual(num_vertices, found_num_vertices)

        # Make sure every vertex has the correct number of hyperedges
        num_hyperedges_dict = {("site1", "site2"): 2, ("site2", "site3"): 2, ("site2", "site4"): 2,
                               ("site1", "site5"): 2, ("site5", "site6"): 2, ("site5", "site7"): 2}
        for edge_id in num_hyperedges_dict.keys():
            found_num_hyperedges = len(state_diagram.vertices[edge_id].hyperedges)
            self.assertEqual(num_hyperedges_dict[edge_id], found_num_hyperedges)

    def test_get_all_vertices(self):
        # Empty -> []
        self.assertEqual([], self.state_diagram_empty.get_all_vertices())

        # Non-Empty
        state_diagram = SingleTermDiagram.from_single_term(
            self.term, self.ref_tree)
        self.assertEqual(6, len(state_diagram.get_all_vertices()))

    def test_get_all_hyperedges(self):
        # Empty -> []
        self.assertEqual([], self.state_diagram_empty.get_all_hyperedges())

        # Non-empty
        state_diagram = SingleTermDiagram.from_single_term(
            self.term, self.ref_tree)
        self.assertEqual(7, len(state_diagram.get_all_hyperedges()))

    def test_get_hyperedge_label(self):
        state_diagram = SingleTermDiagram.from_single_term(
            self.term, self.ref_tree)
        for node_id, label in self.term.items():
            self.assertEqual(label, state_diagram.get_hyperedge_label(node_id))


if __name__ == "__main__":
    unittest.main()
