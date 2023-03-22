class HyperEdge():
    def __init__(self, label, vertices):
        self.label = label
        self.vertices = vertices

class Vertex():
    def __init__(self, hyperedges):
        self.hyperedges = hyperedges

class HyperEdgeColl():
    def __init__(self, corr_node, contained_hyperedges):
        self.corr_node = corr_node
        self.contained_hyperedges = contained_hyperedges

class VertexColl():
    def __init__(self, corr_edge, contained_vertices):
        """


        Parameters
        ----------
        corr_edge : tuple of str
            Contains the identifiers of the two nodes, which are connected by
            the edge corresponding to this collection of vertices.
        contained_vertices : list of Vertex
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.corr_edge = corr_edge
        self.contained_vertices = contained_vertices

class StateDiagram():

    def __init__(self):


        self.vertex_colls = []
        self.hyperedge_colls = []

    @staticmethod
    def from_single_term(self, term, reference_tree):
        """
        Creates a StateDiagram corresponding to a single Hamiltonian term.

        Parameters
        ----------
        term : dict
            The keys are identifiers of the site to which the value, an operator,
            is to be applied.
        reference_tree: TreeTensorNetwork
            Provides the underlying tree structure and the identifiers of all
            nodes.

        Returns
        -------
        state_diag: StateDiagram
        """

        root_id = reference_tree.root_id

        he_root = HyperEdge(term[root_id], None)

        node = reference_tree.nodes[root_id]

        sd = StateDiagram()

        vertices = []
        for child_id in node.children_legs:
            vertex = Vertex([he_root])
            vertices.append(vertex)
            sd.vertex_colls.append(VertexColl((root_id, child_id), [vertex]))

        he_root.vertices = vertices
        sd.hyperedge_colls.append(HyperEdgeColl(root_id, [he_root]))

        for vertex_coll in sd.vertex_colls:
            sd.from_single_term_rec(vertex_coll, term, reference_tree)

    def from_single_term_rec(self, vertex_coll, term, reference_tree):
        new_node_id = vertex_coll.corr_edge[1]
        old_vertex = vertex_coll.contained_vertices[0]

        new_hyperedge = HyperEdge(term[new_node_id], old_vertex)
        old_vertex.hyperedges.append(new_hyperedge)

        new_hyperedge_coll = HyperEdgeColl(new_node_id, [new_hyperedge])

        node = reference_tree.nodes[new_node_id]
        vertices = []
        vertex_colls = []
        for child_id in node.children_legs:
            vertex = Vertex([new_hyperedge])
            vertices.append(vertex)
            vertex_colls.append(VertexColl((new_node_id, child_id), [vertex]))
        self.vertex_colls.extend(vertex_colls)

        new_hyperedge.vertices = vertices
        self.hyperedge_colls.append(HyperEdgeColl(new_node_id, [new_hyperedge]))

        for vertex_col in vertex_colls:
            self.from_single_term_rec(vertex_coll, term, reference_tree)

