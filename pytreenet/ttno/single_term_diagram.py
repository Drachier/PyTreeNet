"""
A state diagram representing a single term Hamiltonian
"""
from typing import Dict, List, Union
from fractions import Fraction

from ..core import TreeStructure
from .hyperedge import HyperEdge
from .vertex import Vertex


class SingleTermDiagram():
    """
    A state diagram representing a single term.

    This single term is a single term of an operator, usually a Hamiltonian,
    that can be represented as a sum of tensor products.
    As this class only represents a single term, all hyperedges and vertices
    are uniquely associated to an object in a reference tree.

    Attributes:
        hyperedges (Dict[str,HyperEdge]): All hyperedges, of which each is
            uniquely associated to a node.
        vertices (Dict[str,Vertex]): All verties, of which each is uniquely
            associated to an edge.
        reference_tree (TreeStructure): A tree structure on which the state
            diagram is based, i.e. both have the same structure.
    """

    def __init__(self, reference_tree: TreeStructure):
        """
        Initialises a SingleTermDiagram.

        There is only a single vertex and a single hyperedge so we don't need
        the collections and save all vertices and edges directly as attributes
        in a dictionary.

        Args:
            reference_tree (TreeStructure): A tree to be used as a reference
                structure.
        """
        self.hyperedges: Dict[str,HyperEdge] = {}
        self.vertices: Dict[str,Vertex] = {}
        self.reference_tree = reference_tree

    def __str__(self) -> str:
        """
        A human-readable string representation.
        """
        string = "Hyperedges: " + str({he.corr_node_id: f"{he.label} ({he.lambda_coeff} * {he.gamma_coeff})" for he in self.hyperedges.values()}) + "\n"
        string += "Vertices: " + str([vert_id for vert_id in self.vertices])
        return string

    def get_all_vertices(self) -> List[Vertex]:
        """
        Returns all vertices of this state diagram.
        """
        return list(self.vertices.values())

    def get_all_hyperedges(self) -> List[HyperEdge]:
        """
        Returns all hyperedges of this state diagram.
        """
        return list(self.hyperedges.values())

    def get_hyperedge_label(self, node_id: str) -> str:
        """
        Returns the label of a hyperedge.

        Args:
            node_id (str): Specifies which hyperedge to return the label from.
        """
        return self.hyperedges[node_id].label

    @classmethod
    def from_single_term(cls,
                         term: Union[tuple[Fraction,str,Dict[str,str]], Dict[str,str]],
                         reference_tree: TreeStructure):
        """
        Creates a state diagram corresponding to a single Hamiltonian term.

        Args:
            term (Dict[str,str]): The term the new state diagram should 
                represent. The leys are identifiers of nodes to which the
                value, a symbolic operator, is to be applied.
            reference_tree (TreeStructure): Provides the underlying tree
                structure and the identifiers of all nodes.

        Returns:
            SingleTermDiagram: The state diagram associated to this single
                term.
        """
        if isinstance(term, dict):
            term = (Fraction(1), "1", term)
        
        assert len(term[2]) == len(reference_tree.nodes), "The term and reference_tree are incompatible!"
        state_diag = cls(reference_tree)
        state_diag._from_single_term_rec((None, reference_tree.root_id), term[2], (term[0], term[1]))
        return state_diag

    def _from_single_term_rec(self, edge_id_tuple, term, coeff_tuple):
        new_node_id = edge_id_tuple[1]  # The old one would be at index 0
        node = self.reference_tree.nodes[new_node_id]

        if not node.is_root():
            old_vertex = self.vertices[edge_id_tuple]
            new_hyperedge = HyperEdge(new_node_id, term[new_node_id], [old_vertex])
            old_vertex.hyperedges.append(new_hyperedge)
        else:  # In this case there is no old vertex
            new_hyperedge = HyperEdge(new_node_id, term[new_node_id], [], coeff_tuple[0], coeff_tuple[1])
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
            self._from_single_term_rec(vertex.corr_edge, term, coeff_tuple)
