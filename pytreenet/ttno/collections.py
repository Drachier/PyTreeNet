"""
This module provides collections/sets of hyperedges and vertices.
"""

from __future__ import annotations
from typing import List, Union, Tuple

import numpy as np

from .hyperedge import HyperEdge
from .vertex import Vertex

class HyperEdgeColl():
    """
    Holds multiple hyperedges that correspond to the same node.

    Attributes:
        corr_node_id (str): The identifier of the node corresponding to this
            collection.
        contained_hyperedges (List[Hyperedge]): A list of all hyperedges
            contained in this collection.
    """

    def __init__(self, corr_node_id: str,
                 contained_hyperedges: Union[None, List[HyperEdge]] = None) -> None:
        """
        Initialises a hyper edge collection.

        Args:
            corr_node_id (str): Identifier of a node to which this collection
                should correspond
            contained_hyperedges (Union[None, List[HyperEdge]]): A list of
                hyperedges that are already included in this collection.
                Default is None
        """
        self.corr_node_id = corr_node_id
        if contained_hyperedges is None:
            contained_hyperedges = []
        self.contained_hyperedges: List[HyperEdge] = contained_hyperedges

    def get_all_labels(self) -> List[str]:
        """
        Obtain all lables of all hyperedges in this collection.

        Returns:
            List[str]: A list of all labels.
        """
        return [he.label for he in self.contained_hyperedges]

    def get_hyperedges_by_label(self, label: str) -> List[HyperEdge]:
        """
        Returns all hyperedges in this collection with a given label.

        Args:
            label (str): The label to be looked for.

        Returns:
            List[HyperEdge]: A list of all hyperedges with the label given.
        """
        return [hyper_edge for hyper_edge in self.contained_hyperedges
                if hyper_edge.label == label]

    def get_completely_contained_hyperedges(self) -> List[HyperEdge]:
        """
        Returns all hyperedges connected only to vertices marked as contained.
        """
        return [hyperedge for hyperedge in self.contained_hyperedges
                if hyperedge.all_vertices_contained()]

    def get_connected_edges(self) -> set[str]:
        """
        Returns a set of all the other nodes connected to this node via edges

        Returns:
            set[str]: A set of all node identifiers connected to this node.
        """
        connected_edges = set()
        if not self.contained_hyperedges:
            return connected_edges
        # Note that all hyperedges should connect to the same set of nodes,
        # so it suffices to just look at the first hyperedge.
        first_he = self.contained_hyperedges[0]
        for vertex in first_he.vertices:
            other = vertex.get_second_node_id(self.corr_node_id)
            connected_edges.add(other)
        return connected_edges

    def physical_dimension(self,
                           conversion_dict: dict[str, np.ndarray]) -> int:
        """
        Returns the physical dimension of this collection.

        Args:
            conversion_dict (dict[str, np.ndarray]): A dictionary mapping labels to
                their operators.

        Returns:
            int: The physical dimension of this collection.
        """
        if not self.contained_hyperedges:
            errstr = "Cannot determine physical dimension of empty "
            errstr += "HyperEdgeColl!"
            raise ValueError(errstr)
        first_he = self.contained_hyperedges[0]
        return conversion_dict[first_he.label].shape[0]

class VertexColl:
    """
    Holds multiple vertices that correspond to the same edge.

    Attributes:
        corr_edge (Tuple[str,str]): The identifiers of the two nodes that are
            connected by the edge corresponding to this collectino of vertices.
        contained_vertices (List[Vertex]): A list of all vertices in this
            collection.
    """

    def __init__(self, corr_edge: Tuple[str, str],
                 contained_vertices: Union[None, List[Vertex]]) -> None:
        """
        Initialises a vertex collection.

        Args:
            corr_edge (Tuple[str, str]): Contains the identifiers of two nodes
                which are connected by the edge corresponding to this
                collection of vertices.
            contained_vertices (Union[None, List[HyperEdge]]): A list of
                vertices already contained in this collection.
        """
        self.corr_edge = corr_edge
        if contained_vertices is None:
            contained_vertices = []
        self.contained_vertices: List[Vertex] = contained_vertices

    def num_vertices(self) -> int:
        """
        Returns the number of vertices in this collection.
        """
        return len(self.contained_vertices)

    def contains_contained(self) -> bool:
        """
        Returns, if any of the vertices in this collection are marked as
        contained.
        """
        for vertex in self.contained_vertices:
            if vertex.contained:
                return True
        return False

    def get_all_marked_vertices(self) -> List[Vertex]:
        """
        Returns all vertices in this collection marked as contained or new.
        """
        return [vertex for vertex in self.contained_vertices
                if vertex.contained or vertex.new]

    def index_vertices(self):
        """
        Indexes all vertices contained in this collection.
        
        This index is the index value to which this vertex corresponds in the
        bond dimension.
        """
        for index, vertex in enumerate(self.contained_vertices):
            vertex.index = index
