"""A state diagram representing a single term Hamiltonian
"""

from .hyperedge import HyperEdge
from .vertex import Vertex


class SingleTermDiagram():
    """A state diagram representing a single term Hamiltonian

    """

    def __init__(self, reference_tree):
        """
        There is only a single vertex and a single hyperedge so
        we don't need the collections and save all vertices
        and edges directly as attributes in a dictionary.
        """
        self.hyperedges = {}
        self.vertices = {}

        self.reference_tree = reference_tree

    def get_all_vertices(self):
        """
        Returns all vertices from all collections in a list.
        """
        return list(self.vertices.values())

    def get_all_hyperedges(self):
        """
        Returns all hyperedges from all collections in a list
        """
        return list(self.hyperedges.values())

    def get_hyperedge_label(self, node_id):
        """
        Returns the label of the hyperedge at 'node_id'.
        """
        return self.hyperedges[node_id].label

    @classmethod
    def from_single_term(cls, term, reference_tree):
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
        assert len(term) == len(reference_tree.nodes), "The term and reference_tree are incompatible!"

        state_diag = cls(reference_tree)
        state_diag._from_single_term_rec((None, reference_tree.root_id), term)

        return state_diag

    def _from_single_term_rec(self, edge_id_tuple, term):
        new_node_id = edge_id_tuple[1]  # The old one would be at index 0
        node = self.reference_tree.nodes[new_node_id]

        if not node.is_root():
            old_vertex = self.vertices[edge_id_tuple]
            new_hyperedge = HyperEdge(new_node_id, term[new_node_id], [old_vertex])
            old_vertex.hyperedges.append(new_hyperedge)
        else:  # In this case there is no old vertex
            new_hyperedge = HyperEdge(new_node_id, term[new_node_id], [])
        self.hyperedges[new_node_id] = new_hyperedge
        if node.is_leaf():  # In this case there is nothing else to do.
            return

        # Initialising all new vertices connected to the hyperedge
        vertices = []  # To update the new_hyperedge later on.
        for child_id in node.children:
            vertex = Vertex((new_node_id, child_id), [new_hyperedge])
            self.vertices[(new_node_id, child_id)] = vertex
            vertices.append(vertex)

        # Updating the new hyperedge with the new vertices
        new_hyperedge.vertices.extend(vertices)

        # Continue recursively below all the new vertices
        for vertex in vertices:
            self._from_single_term_rec(vertex.corr_edge, term)
