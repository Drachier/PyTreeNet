import unittest

import pytreenet as ptn
from pytreenet.ttno import (Vertex,
                            HyperEdge)

class TestStateDiagram(unittest.TestCase):

    def setUp(self) -> None:
        self.node_ids = ("site1", "site2")
        self.vertex_empty = Vertex(self.node_ids, [])

        hyperedges = [HyperEdge("site1", "X", []),
                      HyperEdge("site1", "Y", []),
                      HyperEdge("site2", "Z", []),
                      HyperEdge("site2", "X", [])]
        self.hyperedge_ids = [hyperedge.identifier for hyperedge in hyperedges]
        self.vertex_full = Vertex(self.node_ids, [])
        self.vertex_full.add_hyperedges(hyperedges)

    def test_check_validity_of_node(self):
        self.assertRaises(ValueError, self.vertex_empty.check_validity_of_node, "wrong_node_id")

        self.vertex_empty.check_validity_of_node("site1")
        self.vertex_empty.check_validity_of_node("site2")

    def test_get_hyperedges_for_one_node_id(self):
        # No hyperedges means empty list
        self.assertEqual([], self.vertex_empty.get_hyperedges_for_one_node_id("site1"))
        self.assertEqual([], self.vertex_empty.get_hyperedges_for_one_node_id("site2"))

        # With hyperedges
        for ind, node_id in enumerate(self.node_ids):
            found_hyperedges = self.vertex_full.get_hyperedges_for_one_node_id(node_id)
            self.assertEqual(2, len(found_hyperedges))

            # Check matching ids
            found_ids = [hyperedge.identifier for hyperedge in found_hyperedges]
            expected_ids = self.hyperedge_ids[2*ind:2*(ind +1)]
            self.assertEqual(expected_ids, found_ids)

    def test_hyperedges_to_node(self):
        # No hyperedges
        for node_id in self.node_ids:
            self.assertEqual(0, self.vertex_empty.num_hyperedges_to_node(node_id))

        # With hyperedges
        for node_id in self.node_ids:
            self.assertEqual(2, self.vertex_full.num_hyperedges_to_node(node_id))

    def test_check_hyperedge_uniqueness(self):
        # No hyperedges
        for node_id in self.node_ids:
            self.assertFalse(self.vertex_empty.check_hyperedge_uniqueness(node_id))

        # Too many hyperedges
        for node_id in self.node_ids:
            self.assertFalse(self.vertex_full.check_hyperedge_uniqueness(node_id))

        # One unique, one too much
        vertex_temp = Vertex(self.node_ids, [])
        hyperedges = [HyperEdge("site1", "X", []),
                        HyperEdge("site1", "Y", []),
                        HyperEdge("site2", "Z", [])]
        vertex_temp.add_hyperedges(hyperedges)
        self.assertFalse(vertex_temp.check_hyperedge_uniqueness("site1"))
        self.assertTrue(vertex_temp.check_hyperedge_uniqueness("site2"))

    def test_get_second_node_id(self):
        self.assertRaises(ValueError, self.vertex_empty.get_second_node_id, "wrong_id")
        self.assertEqual("site2", self.vertex_empty.get_second_node_id("site1"))
        self.assertEqual("site1", self.vertex_empty.get_second_node_id("site2"))


if __name__ == "__main__":
    unittest.main()
