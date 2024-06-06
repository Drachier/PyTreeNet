import unittest

import pytreenet as ptn
from pytreenet.ttno import HyperEdge, Vertex

class TestStateDiagram(unittest.TestCase):

    def setUp(self) -> None:
        self.hyperedge_empty = HyperEdge("site1", "X", [])

        vertices = [Vertex(("site1, site2"),[]),
                    Vertex(("site1, site3"),[]),
                    Vertex(("site1, site4"),[])]
        self.vertex_ids = [vertex.identifier for vertex in vertices]
        self.other_node_ids = ["site2", "site3", "site4"]
        self.hyperedge_full = HyperEdge("site1", "X", [])
        for vertex in vertices:
            self.hyperedge_full.add_vertex(vertex)

    def test_find_vertex(self):
        # Throws error for empty he
        self.assertRaises(ValueError, self.hyperedge_empty.find_vertex, "any_node_id")

        # For filled hyperedge
        for ind, other_node_id in enumerate(self.other_node_ids):
            found_vertex_id = self.hyperedge_full.find_vertex(other_node_id).identifier
            expected_vertex_id = self.vertex_ids[ind]
            self.assertEqual(expected_vertex_id, found_vertex_id)

        # Multiple nodes
        self.hyperedge_full.vertices.append(Vertex(("site1", "site2"), [self.hyperedge_full]))
        self.assertRaises(AssertionError, self.hyperedge_full.find_vertex, "site2")

    def test_vertex_single_he(self):
        # No such vertex -> raises error
        self.assertRaises(ValueError, self.hyperedge_full.vertex_single_he, "wrong_node_id")

        # Is unique hyperedge
        self.assertTrue(self.hyperedge_full.vertex_single_he(self.other_node_ids[0]))

        # Not unique hyperedge
        new_hyperedge = HyperEdge("site1", "Z", [])
        new_hyperedge.add_vertex(self.hyperedge_full.vertices[0])
        self.assertFalse(self.hyperedge_full.vertex_single_he(self.other_node_ids[0]))

    def test_get_contained_vertices(self):
        # No vertices -> []
        self.assertEqual([], self.hyperedge_empty.get_contained_vertices())

        # No contained vertices -> empty
        self.assertEqual([], self.hyperedge_full.get_contained_vertices())

        # Once contained vertex
        self.hyperedge_full.vertices[0].contained = True
        found_vertices = self.hyperedge_full.get_contained_vertices()
        self.assertEqual(1, len(found_vertices))
        self.assertEqual(self.vertex_ids[0], found_vertices[0].identifier)

        # All vertices are contained
        for vertex in self.hyperedge_full.vertices:
            vertex.contained = True
        found_vertices = self.hyperedge_full.get_contained_vertices()
        self.assertEqual(3, len(found_vertices))
        self.assertEqual(self.vertex_ids, [vertex.identifier for vertex in found_vertices])

    def test_get_uncontained_vertices(self):
        # No vertices -> []
        self.assertEqual([], self.hyperedge_empty.get_uncontained_vertices())

        # All vertices uncontained
        found_vertices = self.hyperedge_full.get_uncontained_vertices()
        self.assertEqual(3, len(found_vertices))
        self.assertEqual(self.vertex_ids, [vertex.identifier for vertex in found_vertices])

        # All vertices contained -> empty
        for vertex in self.hyperedge_full.vertices:
            vertex.contained = True
        self.assertEqual([], self.hyperedge_full.get_uncontained_vertices())

        # One vertex uncontained
        self.hyperedge_full.vertices[0].contained = False
        found_vertices = self.hyperedge_full.get_uncontained_vertices()
        self.assertEqual(1, len(found_vertices))
        self.assertEqual(self.vertex_ids[0], found_vertices[0].identifier)

    def test_get_single_uncontained_vertex(self):
        # All vertices uncontained
        self.assertRaises(AssertionError, self.hyperedge_full.get_single_uncontained_vertex)

        # All vertices contained -> empty
        for vertex in self.hyperedge_full.vertices:
            vertex.contained = True
        self.assertRaises(AssertionError, self.hyperedge_full.get_single_uncontained_vertex)

        # One vertex uncontained
        self.hyperedge_full.vertices[0].contained = False
        found_vertex = self.hyperedge_full.get_single_uncontained_vertex()
        self.assertEqual(self.vertex_ids[0], found_vertex.identifier)

    def test_num_of_vertices_contained(self):
        # No vertices -> 0
        self.assertEqual(0, self.hyperedge_empty.num_of_vertices_contained())

        # No contained vertices -> 0
        self.assertEqual(0, self.hyperedge_full.num_of_vertices_contained())

        # Once contained vertex
        self.hyperedge_full.vertices[0].contained = True
        self.assertEqual(1, self.hyperedge_full.num_of_vertices_contained())

        # All vertices are contained
        for vertex in self.hyperedge_full.vertices:
            vertex.contained = True
        self.assertEqual(3, self.hyperedge_full.num_of_vertices_contained())
    
    def test_all_vertices_contained(self):
        # No vertices
        self.assertEqual(True, self.hyperedge_empty.all_vertices_contained())

        # No contained vertices
        self.assertEqual(False, self.hyperedge_full.all_vertices_contained())

        # Once contained vertex
        self.hyperedge_full.vertices[0].contained = True
        self.assertEqual(False, self.hyperedge_full.all_vertices_contained())

        # All vertices are contained
        for vertex in self.hyperedge_full.vertices:
            vertex.contained = True
        self.assertEqual(True, self.hyperedge_full.all_vertices_contained())

    def test_all_but_one_vertex_contained(self):
        # No vertices
        self.assertFalse(self.hyperedge_empty.all_but_one_vertex_contained())

        # All vertices uncontained
        self.assertFalse(self.hyperedge_full.all_but_one_vertex_contained())

        # All vertices contained
        for vertex in self.hyperedge_full.vertices:
            vertex.contained = True
        self.assertFalse(self.hyperedge_full.all_but_one_vertex_contained())

        # Exactly one vertex uncontained
        self.hyperedge_full.vertices[0].contained = False
        self.assertTrue(self.hyperedge_full.all_but_one_vertex_contained())

        # With single vertex
        hyperedge_single = HyperEdge("site1", "X", [Vertex(("site1","site2"), [])])
        self.assertTrue(hyperedge_single.all_but_one_vertex_contained())
        hyperedge_single.vertices[0].contained = True
        self.assertFalse(hyperedge_single.all_but_one_vertex_contained())

if __name__ == "__main__":
    unittest.main()
