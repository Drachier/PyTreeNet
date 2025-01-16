"""
This module provides a hyperedge to be used in state diagrams.
"""
from __future__ import annotations
from typing import List, Tuple, Union
import hashlib
import uuid
from fractions import Fraction

from ..core.tree_structure import TreeStructure

class HyperEdge():
    """
    A hyperedge, i.e. an edge connecting any number of vertices.

    Attributes:
        corr_node_id (str): The node to which this hyperedge corresponds.
        label (str): The label associated to this hyperedge.
        vertices (List[Vertex]): A list of vertices connected to this
            hyperedge.
        hash (str): A hash value corresponding to this hyperedge's label.
        identifier (str): A unique identifier of this hyperedge.

    """
    def __init__(self,
                 corr_node_id: str,
                 label: str,
                 vertices: List[Vertex],
                 lambda_coeff: Fraction = Fraction(1),
                 gamma_coeff: str = "1"):
        """
        Initialises a hyperedge.

        Args:
            corr_node_id (str): The node to which this hyperedge corresponds.
            label (str): The label associated to this hyperedge.
            vertices (List[Vertex]): A list of vertices connected to this
                hyperedge.
        """
        self.corr_node_id = corr_node_id
        self.label = label
        self.vertices = vertices
        self.lambda_coeff = lambda_coeff
        self.gamma_coeff = gamma_coeff

        self.hash = hashlib.sha256(self.label.encode()).hexdigest()
        self.v_hash = None

        self.identifier = str(uuid.uuid1())

    def __repr__(self) -> str:
        """
        A string representation of this hyperedge.
        """
        string = f"label = {self.label}; "
        string += f"corr_site = {self.corr_node_id}; "
        string += "coeff = " + str(self.lambda_coeff) + " * " + str(self.gamma_coeff) + "; "


        string += "connected to "
        for vertex in self.vertices:
            string += f"{str(vertex.corr_edge)}, "
        return string

    def __eq__(self, other_he: HyperEdge) -> bool:
        """
        Implements equality of two hyperedges.

        Two hyperedges are equal, if they have the same label and correspond
        to the same node.

        Args:
            other_he (HyperEdge): The hyperedge to compare to.

        Returns:
            bool: Equality of the two hyperedges
        """
        labels_eq = self.label == other_he.label
        corr_node_id_eq = self.corr_node_id == other_he.corr_node_id
        return labels_eq and corr_node_id_eq

    def __hash__(self) -> int:
        """
        Generate a hash value of the Hyperedge
        """
        return hash((frozenset(self.vertices), self.corr_node_id, self.label))

    def calculate_hash(self,
                       children_hash: Union[str,None] = None) -> str:
        """
        Calculate the hash of the hyperedge.

        Args:
            children_hash (Union[str,None], optional): The hash of vertices
                connected to the hyperedge. Defaults to None.
        """
        hash_text = self.label + children_hash
        self.hash = hashlib.sha256(hash_text.encode()).hexdigest()
        return self.hash

    def get_hash(self) -> str:
        """
        Return the hash of the hyperedge
        """
        return self.hash

    def set_hash(self, hash_val: str):
        """
        Set the hash of the hyperedge
        """
        self.hash = hash_val

    def add_vertex(self, vertex: Vertex):
        """
        Adds a vertex to this hyperedge.

        In turn this hyperedge is also added to the vertex.
        """
        self.vertices.append(vertex)
        vertex.hyperedges.append(self)

    def add_vertices(self, vertices: list):
        """
        Add multiple vertices to this hyperedge.

        In turn this hyperedge is added to all vertices.
        """
        self.vertices.extend(vertices)
        for vertex in vertices:
            vertex.hyperedges.append(self)

    def find_vertex(self, other_node_id: str) -> Vertex:
        """
        Finds the vertex connected to this hyperedge corresponding to the edge 
        (corr_node_id, other_node_id).

        Args:
            other_node_id (str): The vertex corresponds to an edge in the
                underlying tree structure. One node the edge is connected to is
                the node this hyperedge corresponds to, while the other node
                identifier is provided by this string.

        Raises:
            ValueError: The hyperedge is not connected to a fitting vertex.

        Returns:
            Vertex: The vertex corresponding to the edge connecting this
                hyperedge and a hyperedge corresponding to the other specified
                node.
        """
        vertex_list = [vertex for vertex in self.vertices
                       if other_node_id in vertex.corr_edge]
        if len(vertex_list) == 0:
            err_str = f"Hyperedge not connected to a vertex corresponding to edge {(self.corr_node_id, other_node_id)}!"
            raise ValueError(err_str)
        assert_str = "Hyperedge should only be connected to one vertex corresponding to an edge!"
        assert len(vertex_list) == 1, assert_str
        return vertex_list[0]

    def vertex_single_he(self, other_node_id: str) -> bool:
        """
        Checks if a specified vertex is connected to only a single hyperedge.

        The vertex checked, is the vertex connected to this hyperedge and
        a hypergraph corresponding to the other specified node. This method
        checks, if the current hypergraph is the only hypergraph connected to
        that vertex corresponding to this node

        Args:
            other_node_id (str): The vertex corresponds to an edge in the
                underlying tree structure. One node the edge is connected to is
                the node this hyperedge corresponds to, while the other node
                identifier is provided by this string.

        Returns:
            bool: Whether the specified vertex is connected to only a single
                hyperedge.
        """
        vertex = self.find_vertex(other_node_id)
        return vertex.check_hyperedge_uniqueness(self.corr_node_id)

    def get_contained_vertices(self) -> List[Vertex]:
        """
        Returns all vertices marked as contained connected to this hyperedge.
        """
        return [vertex for vertex in self.vertices if vertex.contained]

    def get_uncontained_vertices(self) -> List[Vertex]:
        """
        Returns all vertices not marked as contained connected to this hyperedge.
        """
        return [vertex for vertex in self.vertices if not vertex.contained]

    def get_single_uncontained_vertex(self) -> Vertex:
        """
        Returns a single uncontained vertex.

        Raises:
            AssertionError: If there are more or less than one uncontained
                vertex.
        """
        uncontained_vertices = self.get_uncontained_vertices()
        err_string = "The hyperedge is not connected to a single contained vertex."
        assert len(uncontained_vertices) == 1, err_string
        return uncontained_vertices[0]

    def num_of_vertices_contained(self) -> int:
        """
        The number of vertices connected to this hyperedge marked as contained.

        Returns:
            int: The number of connected contained vertices.
        """
        marked_contained = self.get_contained_vertices()
        num_marked_contained = len(marked_contained)
        return num_marked_contained

    def all_vertices_contained(self) -> bool:
        """
        Returns if all vertices of this hyperedge are marked as contained.
        """
        return len(self.get_uncontained_vertices()) == 0

    def all_but_one_vertex_contained(self) -> bool:
        """
        Returns if all but one vertex of this hyperedge are marked as contained.
        """
        return len(self.get_uncontained_vertices()) == 1

    def find_tensor_position(self,
                             reference_tree: TreeStructure) -> Tuple:
        """
        Finds the position of the operator in the total TTNO tensor.

        The position is of the operator represented by this hyperedge.

        Args:
            reference_tree (TreeStructure): A tree to provide the topology.

        Returns:
            Tuple: The position of the operator. The last two entries are
                slices ":" corresponding to the physical dimensions.
        """
        position = [0] * len(self.vertices)
        position.extend([slice(None), slice(None)])
        for vertex in self.vertices:
            other_node_id = vertex.get_second_node_id(self.corr_node_id)
            index_position = reference_tree.nodes[self.corr_node_id].neighbour_index(other_node_id)
            position[index_position] = vertex.index
        return tuple(position)
    
    def calculate_v_hash(self, current_node, parent):
        """
        Calculates the hash of the hyperedge for combining as v nodes.
        """
        hash_text = self.label + len(self.vertices).__str__() 

        vertices = []
        for v in self.vertices:
            if not(v.corr_edge == (current_node,parent) or v.corr_edge == (parent,current_node)):
                vertices.append(v.identifier)
        
        vertices.sort()
        hash_text += "".join(vertices)

        self.v_hash = hashlib.sha256(hash_text.encode()).hexdigest() 
        return self.v_hash
    
