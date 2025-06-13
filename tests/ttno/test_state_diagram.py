import unittest

import pytreenet as ptn
from pytreenet.random import random_tensor_node
from pytreenet.ttno import (HyperEdge,
                            StateDiagram)

def check_hyperedge_coll(state_diagram, node_id, he_labels, num_hes, num_connected_vertices):
    """
    Checks a HyperedgeColl and less exact the he in it. More precisely
    the number of hyperedges contained are checked and for each hyperedge
    it is checked, if the corresponding node_id is correct, the label is
    one of a few possible labels given by he_labels and if the number of
    vertices the hyperedge is connected to is one possible number given
    by the set num_connected_vertices.

    Parameters
    ----------
    node_id: str
        The node_id identifiying the Hyp9
        eredgeColl to check
    he_labels: set
        A set of possible lables the he can have
    num_hes: int
        The number of hyperedges which are supposed to be in the
        HyperedgeColl
    num_connected_vertices: set of int
        The possibe number of vertices the hyperedges can be connected to

    Returns
    -------
    None

    """
    he_coll = state_diagram.hyperedge_colls[node_id]
    assert len(he_coll.contained_hyperedges) == num_hes
    for he in he_coll.contained_hyperedges:
        assert he.corr_node_id == node_id
        assert he.label in he_labels
        assert len(he.vertices) in num_connected_vertices


def check_vertex_coll(state_diagram, corr_edge, num_vertices, num_connected_hes):
    """
    Checks a VertexColl and less exact the vertices in it. More precisely
    the number of vertices contained are checked and for each vertex
    it is checked, if the number of hyperedges the vertex is connected to
    is one possible number given by the set num_connected_hes.

    Parameters
    ----------
    corr_edge: tuple of str
        The edge identifiying the VertexColl to check
    num_vertices: int
        The number of vertices which are supposed to be in the
        VertexColl
    num_connected_hes: set of int
        The possibe number of hyperedges the vertices can be connected to

    Returns
    -------
    None

    """
    vertex_coll = state_diagram.vertex_colls[corr_edge]
    assert len(vertex_coll.contained_vertices) == num_vertices
    for vertex in vertex_coll.contained_vertices:
        assert len(vertex.hyperedges) in num_connected_hes


class TestonSingleStateDiagram(unittest.TestCase):
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
        self.sd = StateDiagram.from_single_term(self.term, self.ref_tree)

    def reset_check(self):
        # Checking the markers are reset correctly
        for vertex_coll in self.sd.vertex_colls.values():
            for vertex in vertex_coll.contained_vertices:
                self.assertTrue(not vertex.contained)
                self.assertTrue(not vertex.new)

    def test_add_single_term_all_different(self):
        # Building intial sd
        term2 = {"site1": "12", "site2": "22", "site3": "32",
                 "site4": "42", "site5": "52", "site6": "62", "site7": "72"}
        self.sd.add_single_term(term2)

        self.reset_check()

        # The number of of vertex and hyperedge collections depends on the underlying tree.
        self.assertTrue(len(self.sd.hyperedge_colls) == 7)
        self.assertTrue(len(self.sd.vertex_colls) == 6)

        # Every site has two corresponding hyperedges
        for hyperedge_coll in self.sd.hyperedge_colls.values():
            self.assertTrue(len(hyperedge_coll.contained_hyperedges) == 2)

        # Every edge has two corresponding vertices
        for vertex_coll in self.sd.vertex_colls.values():
            self.assertEqual(len(vertex_coll.contained_vertices), 2)

    def check_hyperedge_coll(self, node_id, he_labels, num_hes, num_connected_vertices):
        """
        Checks a HyperedgeColl and less exact the he in it. More precisely
        the number of hyperedges contained are checked and for each hyperedge
        it is checked, if the corresponding node_id is correct, the label is
        one of a few possible labels given by he_labels and if the number of
        vertices the hyperedge is connected to is one possible number given
        by the set num_connected_vertices.

        Parameters
        ----------
        node_id: str
            The node_id identifiying the HyperedgeColl to check
        he_labels: set
            A set of possible lables the he can have
        num_hes: int
            The number of hyperedges which are supposed to be in the
            HyperedgeColl
        num_connected_vertices: set of int
            The possibe number of vertices the hyperedges can be connected to

        Returns
        -------
        None

        """
        he_coll = self.sd.hyperedge_colls[node_id]
        self.assertEqual(len(he_coll.contained_hyperedges), num_hes)
        for he in he_coll.contained_hyperedges:
            self.assertEqual(he.corr_node_id, node_id)
            self.assertTrue(he.label in he_labels)
            self.assertTrue(len(he.vertices) in num_connected_vertices)

    def check_vertex_coll(self, corr_edge, num_vertices, num_connected_hes):
        """
        Checks a VertexColl and less exact the vertices in it. More precisely
        the number of vertices contained are checked and for each vertex
        it is checked, if the number of hyperedges the vertex is connected to
        is one possible number given by the set num_connected_hes.

        Parameters
        ----------
        corr_edge: tuple of str
            The edge identifiying the VertexColl to check
        num_vertices: int
            The number of vertices which are supposed to be in the
            VertexColl
        num_connected_hes: set of int
            The possibe number of hyperedges the vertices can be connected to

        Returns
        -------
        None

        """
        vertex_coll = self.sd.vertex_colls[corr_edge]
        self.assertEqual(len(vertex_coll.contained_vertices), num_vertices)
        for vertex in vertex_coll.contained_vertices:
            self.assertTrue(len(vertex.hyperedges) in num_connected_hes)

    def test_add_single_term_one_leaf_same(self):
        # Building intial sd where the label of site3 is the same as in the original term
        term2 = {"site1": "12", "site2": "22", "site3": "3",
                 "site4": "42", "site5": "52", "site6": "62", "site7": "72"}
        self.sd.add_single_term(term2)

        self.reset_check()

        # Lables and number of connected vertices hes at each site can have
        potential_labels_dict = {"site1": {"1", "12"},
                                 "site2": {"2", "22"},
                                 "site3": {"3"},
                                 "site4": {"4", "42"},
                                 "site5": {"5", "52"},
                                 "site6": {"6", "62"},
                                 "site7": {"7", "72"}}
        potential_num_vertices_dict = {"site1": {2},
                                       "site2": {3},
                                       "site3": {1},
                                       "site4": {1},
                                       "site5": {3},
                                       "site6": {1},
                                       "site7": {1}}

        for node_id, he_coll in self.sd.hyperedge_colls.items():

            if node_id == "site3":
                # Site 3 has only one corresponding hyperedge
                self.assertTrue(len(he_coll.contained_hyperedges) == 1)
                for he in he_coll.contained_hyperedges:
                    self.assertEqual(HyperEdge("site3", "3", []), he)
                    self.assertEqual(1, len(he.vertices))

            else:
                self.check_hyperedge_coll(node_id,
                                          potential_labels_dict[node_id],
                                          2,
                                          potential_num_vertices_dict[node_id])

        for edge in self.sd.vertex_colls:
            if ("site2" in edge) and ("site3" in edge):
                self.check_vertex_coll(edge, 1, {3})
            else:
                self.check_vertex_coll(edge, 2, {2})

    def test_add_single_term_only_one_leaf_different(self):
        term2 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "72"}
        self.sd.add_single_term(term2)

        # There should only be one site with two hyperedges, which is the leaf
        # site
        for s in range(1, 8):
            node_id = "site" + str(s)
            if s == 7:
                self.check_hyperedge_coll(node_id, {"7", "72"}, 2, {1})
            else:
                self.check_hyperedge_coll(node_id, {str(s)}, 1, {1, 2, 3})

    def test_add_single_term_only_root_differen(self):
        # Building intial sd where only the root lable is different
        term2 = {"site1": "12", "site2": "2", "site3": "3",
                 "site4": "4", "site5": "5", "site6": "6", "site7": "7"}
        self.sd.add_single_term(term2)

        self.reset_check()

        # The root should habe two hyperedges corresponding to it, but they
        # connect to the same vertices. The remainig sites have only one he
        # corresponding to them. When going to the TTNO the two hes at the root
        # would be added together.
        potential_num_vertices_dict = {"site1": {},
                                       "site2": {3},
                                       "site3": {1},
                                       "site4": {1},
                                       "site5": {3},
                                       "site6": {1},
                                       "site7": {1}}
        for i in range(1, 8):
            node_id = "site" + str(i)
            if i == 1:
                self.check_hyperedge_coll(node_id, {"1", "12"}, 2, {2})
            else:
                self.check_hyperedge_coll(node_id, {str(i)}, 1,
                                          potential_num_vertices_dict[node_id])

        for edge in self.sd.vertex_colls:
            if ("site1" in edge):
                self.check_vertex_coll(edge, 1, {3})
            else:
                self.check_vertex_coll(edge, 1, {2})

    def test_add_single_term_same_up_to_root(self):
        # Building additional sd where one subtree of the root and the root itself is the same
        term2 = {"site1": "1",
                 "site2": "22",
                 "site3": "32",
                 "site4": "42",
                 "site5": "5",
                 "site6": "6",
                 "site7": "7"}
        self.sd.add_single_term(term2)

        # The subtree from 2 down, should contain two paths, the remaining
        # state diagram should have unique hyperedges and vertices
        potential_num_vertices_dict = {"site1": {2},
                                       "site2": {3},
                                       "site3": {1},
                                       "site4": {1},
                                       "site5": {3},
                                       "site6": {1},
                                       "site7": {1}}

        potential_labels_dict = {"site1": {"1"},
                                 "site2": {"2", "22"},
                                 "site3": {"3", "32"},
                                 "site4": {"4", "42"},
                                 "site5": {"5"},
                                 "site6": {"6"},
                                 "site7": {"7"}}

        for i in range(1, 8):
            node_id = "site" + str(i)
            self.check_hyperedge_coll(node_id,
                                      potential_labels_dict[node_id],
                                      len(potential_labels_dict[node_id]),
                                      potential_num_vertices_dict[node_id])

        potential_num_connected_he = {('site1', 'site2'): {3},
                                      ('site1', 'site5'): {2},
                                      ('site2', 'site3'): {2},
                                      ('site2', 'site4'): {2},
                                      ('site5', 'site6'): {2},
                                      ('site5', 'site7'): {2}}

        potential_num_vertices = {('site1', 'site2'): 1,
                                  ('site1', 'site5'): 1,
                                  ('site2', 'site3'): 2,
                                  ('site2', 'site4'): 2,
                                  ('site5', 'site6'): 1,
                                  ('site5', 'site7'): 1}

        for edge in self.sd.vertex_colls:
            self.check_vertex_coll(edge,
                                   potential_num_vertices[edge],
                                   potential_num_connected_he[edge])

    def test_add_single_term_on_different_in_middle(self):
        # Building additional sd where one subtree of the root and the root itself is the same
        term2 = {"site1": "1",
                 "site2": "22",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "7"}
        self.sd.add_single_term(term2)

        # The subtree from 2 down, should contain two paths, the remaining
        # state diagram should have unique hyperedges and vertices
        potential_num_vertices_dict = {"site1": {2},
                                       "site2": {3},
                                       "site3": {1},
                                       "site4": {1},
                                       "site5": {3},
                                       "site6": {1},
                                       "site7": {1}}

        potential_labels_dict = {"site1": {"1"},
                                 "site2": {"2", "22"},
                                 "site3": {"3"},
                                 "site4": {"4"},
                                 "site5": {"5"},
                                 "site6": {"6"},
                                 "site7": {"7"}}

        for i in range(1, 8):
            node_id = "site" + str(i)
            self.check_hyperedge_coll(node_id,
                                      potential_labels_dict[node_id],
                                      len(potential_labels_dict[node_id]),
                                      potential_num_vertices_dict[node_id])

        potential_num_connected_he = {('site1', 'site2'): {3},
                                      ('site1', 'site5'): {2},
                                      ('site2', 'site3'): {3},
                                      ('site2', 'site4'): {3},
                                      ('site5', 'site6'): {2},
                                      ('site5', 'site7'): {2}}

        potential_num_vertices = {('site1', 'site2'): 1,
                                  ('site1', 'site5'): 1,
                                  ('site2', 'site3'): 1,
                                  ('site2', 'site4'): 1,
                                  ('site5', 'site6'): 1,
                                  ('site5', 'site7'): 1}

        for edge in self.sd.vertex_colls:
            self.check_vertex_coll(edge,
                                   potential_num_vertices[edge],
                                   potential_num_connected_he[edge])


class TestFromHamiltonian(unittest.TestCase):
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

    def check_hyperedge_coll(self, state_diagram, node_id, he_labels, num_hes, num_connected_vertices):
        """
        Checks a HyperedgeColl and less exact the he in it. More precisely
        the number of hyperedges contained are checked and for each hyperedge
        it is checked, if the corresponding node_id is correct, the label is
        one of a few possible labels given by he_labels and if the number of
        vertices the hyperedge is connected to is one possible number given
        by the set num_connected_vertices.

        Parameters
        ----------
        node_id: str
            The node_id identifiying the HyperedgeColl to check
        he_labels: set
            A set of possible lables the he can have
        num_hes: int
            The number of hyperedges which are supposed to be in the
            HyperedgeColl
        num_connected_vertices: set of int
            The possibe number of vertices the hyperedges can be connected to

        Returns
        -------
        None

        """
        he_coll = state_diagram.hyperedge_colls[node_id]
        self.assertEqual(len(he_coll.contained_hyperedges), num_hes)
        for he in he_coll.contained_hyperedges:
            self.assertEqual(he.corr_node_id, node_id)
            self.assertTrue(he.label in he_labels)
            self.assertTrue(len(he.vertices) in num_connected_vertices)

    def check_vertex_coll(self, state_diagram, corr_edge, num_vertices, num_connected_hes):
        """
        Checks a VertexColl and less exact the vertices in it. More precisely
        the number of vertices contained are checked and for each vertex
        it is checked, if the number of hyperedges the vertex is connected to
        is one possible number given by the set num_connected_hes.

        Parameters
        ----------
        corr_edge: tuple of str
            The edge identifiying the VertexColl to check
        num_vertices: int
            The number of vertices which are supposed to be in the
            VertexColl
        num_connected_hes: set of int
            The possibe number of hyperedges the vertices can be connected to

        Returns
        -------
        None

        """
        vertex_coll = state_diagram.vertex_colls[corr_edge]
        self.assertEqual(len(vertex_coll.contained_vertices), num_vertices)
        for vertex in vertex_coll.contained_vertices:
            self.assertTrue(len(vertex.hyperedges) in num_connected_hes)

    def test_from_hamiltonian_single_term(self):
        term1 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "7"}
        terms = [term1]
        hamiltonian = ptn.Hamiltonian(terms=terms)

        sd = StateDiagram.from_hamiltonian(hamiltonian, self.ref_tree)

        self.assertEqual(6, len(sd.vertex_colls))
        self.assertEqual(7, len(sd.hyperedge_colls))

        potential_labels_dict = {"site1": {"1"},
                                 "site2": {"2"},
                                 "site3": {"3"},
                                 "site4": {"4"},
                                 "site5": {"5"},
                                 "site6": {"6"},
                                 "site7": {"7"}}

        potential_num_vertices_dict = {"site1": {2},
                                       "site2": {3},
                                       "site3": {1},
                                       "site4": {1},
                                       "site5": {3},
                                       "site6": {1},
                                       "site7": {1}}

        for edge in sd.vertex_colls:
            self.check_vertex_coll(sd, edge, 1, {2})
        for node_id in sd.hyperedge_colls:
            self.check_hyperedge_coll(sd, node_id,
                                      potential_labels_dict[node_id],
                                      1, potential_num_vertices_dict[node_id])

    def test_from_hamiltonian_two_terms(self):
        term1 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "7"}
        term2 = {"site1": "12",
                 "site2": "22",
                 "site3": "32",
                 "site4": "42",
                 "site5": "52",
                 "site6": "62",
                 "site7": "72"}
        terms = [term1, term2]
        hamiltonian = ptn.Hamiltonian(terms=terms)

        sd = StateDiagram.from_hamiltonian(hamiltonian, self.ref_tree)

        # The number of of vertex and hyperedge collections depends on the underlying tree.
        self.assertTrue(len(sd.hyperedge_colls) == 7)
        self.assertTrue(len(sd.vertex_colls) == 6)

        # Every site has two corresponding hyperedges
        for hyperedge_coll in sd.hyperedge_colls.values():
            self.assertTrue(
                len(hyperedge_coll.contained_hyperedges) == 2)

        # Every edge has two corresponding vertices
        for vertex_coll in sd.vertex_colls.values():
            self.assertEqual(len(vertex_coll.contained_vertices), 2)

    def test_from_hamiltonian_three_terms(self):
        term1 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "7"}
        term2 = {"site1": "12",
                 "site2": "22",
                 "site3": "32",
                 "site4": "42",
                 "site5": "52",
                 "site6": "62",
                 "site7": "72"}
        term3 = {"site1": "13",
                 "site2": "23",
                 "site3": "33",
                 "site4": "43",
                 "site5": "53",
                 "site6": "63",
                 "site7": "73"}

        terms = [term1, term2, term3]
        hamiltonian = ptn.Hamiltonian(terms=terms)

        sd = StateDiagram.from_hamiltonian(hamiltonian, self.ref_tree)

        # The number of of vertex and hyperedge collections depends on the underlying tree.
        self.assertTrue(len(sd.hyperedge_colls) == 7)
        self.assertTrue(len(sd.vertex_colls) == 6)

        potential_num_vertices_dict = {"site1": {2},
                                       "site2": {3},
                                       "site3": {1},
                                       "site4": {1},
                                       "site5": {3},
                                       "site6": {1},
                                       "site7": {1}}
        potential_labels_dict = {"site1": {"1", "12", "13"},
                                 "site2": {"2", "22", "23"},
                                 "site3": {"3", "32", "33"},
                                 "site4": {"4", "42", "43"},
                                 "site5": {"5", "52", "53"},
                                 "site6": {"6", "62", "63"},
                                 "site7": {"7", "72", "73"}}

        for node_id in sd.hyperedge_colls:
            self.check_hyperedge_coll(sd, node_id,
                                      potential_labels_dict[node_id],
                                      3, potential_num_vertices_dict[node_id])

        for edge in sd.vertex_colls:
            self.check_vertex_coll(sd, edge, 3, {2})

    def test_from_hamiltonian_more_complicated_example1(self):
        term1 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "7"}
        term2 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "72"}
        term3 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "53",
                 "site6": "6",
                 "site7": "73"}

        terms = [term1, term2, term3]
        hamiltonian = ptn.Hamiltonian(terms=terms)

        sd = StateDiagram.from_hamiltonian(hamiltonian, self.ref_tree)

        # The number of of vertex and hyperedge collections depends on the underlying tree.
        self.assertTrue(len(sd.hyperedge_colls) == 7)
        self.assertTrue(len(sd.vertex_colls) == 6)

        potential_num_vertices_dict = {"site1": {2},
                                       "site2": {3},
                                       "site3": {1},
                                       "site4": {1},
                                       "site5": {3},
                                       "site6": {1},
                                       "site7": {1}}
        potential_labels_dict = {"site1": {"1"},
                                 "site2": {"2"},
                                 "site3": {"3"},
                                 "site4": {"4"},
                                 "site5": {"5", "53"},
                                 "site6": {"6"},
                                 "site7": {"7", "72", "73"}}

        num_he_dict = {"site1": 1,
                       "site2": 1,
                       "site3": 1,
                       "site4": 1,
                       "site5": 2,
                       "site6": 1,
                       "site7": 3}

        for node_id in sd.hyperedge_colls:
            self.check_hyperedge_coll(sd, node_id,
                                      potential_labels_dict[node_id],
                                      num_he_dict[node_id],
                                      potential_num_vertices_dict[node_id])

        potential_num_connected_he = {('site1', 'site2'): {2},
                                      ('site1', 'site5'): {3},
                                      ('site2', 'site3'): {2},
                                      ('site2', 'site4'): {2},
                                      ('site5', 'site6'): {3},
                                      ('site5', 'site7'): {2, 3}}

        potential_num_vertices = {('site1', 'site2'): 1,
                                  ('site1', 'site5'): 1,
                                  ('site2', 'site3'): 1,
                                  ('site2', 'site4'): 1,
                                  ('site5', 'site6'): 1,
                                  ('site5', 'site7'): 2}

        for edge in sd.vertex_colls:
            self.check_vertex_coll(sd, edge,
                                   potential_num_vertices[edge],
                                   potential_num_connected_he[edge])

    def test_from_hamiltonian_more_complicated_example2(self):
        term1 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "7"}
        term2 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "72"}
        term3 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "4",
                 "site5": "53",
                 "site6": "6",
                 "site7": "73"}
        term4 = {"site1": "1",
                 "site2": "2",
                 "site3": "3",
                 "site4": "44",
                 "site5": "53",
                 "site6": "6",
                 "site7": "73"}

        terms = [term1, term2, term3, term4]
        hamiltonian = ptn.Hamiltonian(terms=terms)

        sd = StateDiagram.from_hamiltonian(hamiltonian, self.ref_tree)

        potential_num_vertices_dict = {"site1": {2},
                                       "site2": {3},
                                       "site3": {1},
                                       "site4": {1},
                                       "site5": {3},
                                       "site6": {1},
                                       "site7": {1}}
        potential_labels_dict = {"site1": {"1"},
                                 "site2": {"2"},
                                 "site3": {"3"},
                                 "site4": {"4", "44"},
                                 "site5": {"5", "53"},
                                 "site6": {"6"},
                                 "site7": {"7", "72", "73"}}

        num_he_dict = {"site1": 2,
                       "site2": 2,
                       "site3": 1,
                       "site4": 2,
                       "site5": 3,
                       "site6": 1,
                       "site7": 3}

        for node_id in sd.hyperedge_colls:
            self.check_hyperedge_coll(sd, node_id,
                                      potential_labels_dict[node_id],
                                      num_he_dict[node_id],
                                      potential_num_vertices_dict[node_id])

        potential_num_connected_he = {('site1', 'site2'): {2},
                                      ('site1', 'site5'): {2, 3},
                                      ('site2', 'site3'): {3},
                                      ('site2', 'site4'): {2},
                                      ('site5', 'site6'): {4},
                                      ('site5', 'site7'): {3}}

        num_vertices = {('site1', 'site2'): 2,
                        ('site1', 'site5'): 2,
                        ('site2', 'site3'): 1,
                        ('site2', 'site4'): 2,
                        ('site5', 'site6'): 1,
                        ('site5', 'site7'): 2}

        for edge in sd.vertex_colls:
            self.check_vertex_coll(sd, edge,
                                   num_vertices[edge],
                                   potential_num_connected_he[edge])


class TestFromHamiltonianAsymmetric(unittest.TestCase):

    def setUp(self):
        self.ref_tree = ptn.TreeTensorNetwork()

        node1, tensor1 = random_tensor_node((2, 2, 2), identifier="site1")
        node2, tensor2 = random_tensor_node((2, 2, 2, 2), identifier="site2")
        node5, tensor5 = random_tensor_node((2, 2, 2, 2), identifier="site5")
        node3, tensor3 = random_tensor_node((2, 2), identifier="site3")
        node4, tensor4 = random_tensor_node((2, 2), identifier="site4")
        node6, tensor6 = random_tensor_node((2, 2), identifier="site6")
        node7, tensor7 = random_tensor_node((2, 2, 2), identifier="site7")
        node8, tensor8 = random_tensor_node((2, 2), identifier="site8")

        self.ref_tree.add_root(node1, tensor1)
        self.ref_tree.add_child_to_parent(node2, tensor2, 0, "site1", 0)
        self.ref_tree.add_child_to_parent(node5, tensor5, 0, "site1", 1)
        self.ref_tree.add_child_to_parent(node3, tensor3, 0, "site2", 1)
        self.ref_tree.add_child_to_parent(node4, tensor4, 0, "site2", 2)
        self.ref_tree.add_child_to_parent(node6, tensor6, 0, "site5", 1)
        self.ref_tree.add_child_to_parent(node7, tensor7, 0, "site5", 2)
        self.ref_tree.add_child_to_parent(node8, tensor8, 0, "site7", 1)

    def test_four_sites(self):
        """
        In this case it can happen, that two vertices at ("site5","site6") are marked as
        contained, which leads to wrong results.
        """

        terms = [{'site3': 'I', 'site6': 'Y', 'site1': 'I', 'site2': 'I',
                  'site4': 'I', 'site5': 'I', 'site7': 'I', 'site8': 'I'},
                 {'site4': 'I', 'site5': 'Z', 'site3': 'Z', 'site6': 'Z',
                  'site8': 'Z', 'site7': 'X', 'site1': 'I', 'site2': 'X'},
                 {'site6': 'I', 'site4': 'I', 'site2': 'Z', 'site8': 'Z',
                  'site7': 'Y', 'site1': 'Y', 'site5': 'Y', 'site3': 'I'},
                 {'site1': 'I', 'site2': 'I', 'site3': 'I', 'site4': 'I',
                  'site5': 'I', 'site6': 'I', 'site7': 'I', 'site8': 'I'}]

        hamiltonian = ptn.Hamiltonian(terms=terms)

        sd = StateDiagram.from_hamiltonian(hamiltonian, self.ref_tree)

        potential_num_vertices_dict = {"site1": {2},
                                       "site2": {3},
                                       "site3": {1},
                                       "site4": {1},
                                       "site5": {3},
                                       "site6": {1},
                                       "site7": {2},
                                       "site8": {1}}
        potential_labels_dict = {"site1": {"I", "Y"},
                                 "site2": {"I", "X", "Z"},
                                 "site3": {"I", "Z"},
                                 "site4": {"I"},
                                 "site5": {"I", "Z", "Y"},
                                 "site6": {"I", "Y", "Z"},
                                 "site7": {"I", "X", "Y"},
                                 "site8": {"I", "Z"}}

        num_he_dict = {"site1": 3,
                       "site2": 3,
                       "site3": 2,
                       "site4": 1,
                       "site5": 4,
                       "site6": 3,
                       "site7": 3,
                       "site8": 2}

        for node_id in sd.hyperedge_colls:
            check_hyperedge_coll(sd, node_id,
                                 potential_labels_dict[node_id],
                                 num_he_dict[node_id],
                                 potential_num_vertices_dict[node_id])

        potential_num_connected_he = {('site1', 'site2'): {2},
                                      ('site1', 'site5'): {2, 3},
                                      ('site2', 'site3'): {2, 3},
                                      ('site2', 'site4'): {4},
                                      ('site5', 'site6'): {2, 3},
                                      ('site5', 'site7'): {2, 3},
                                      ('site7', 'site8'): {2, 3}}

        num_vertices = {('site1', 'site2'): 3,
                        ('site1', 'site5'): 3,
                        ('site2', 'site3'): 2,
                        ('site2', 'site4'): 1,
                        ('site5', 'site6'): 3,
                        ('site5', 'site7'): 3,
                        ('site7', 'site8'): 2}

        for edge in sd.vertex_colls:
            check_vertex_coll(sd, edge,
                              num_vertices[edge],
                              potential_num_connected_he[edge])


if __name__ == "__main__":
    unittest.main()
