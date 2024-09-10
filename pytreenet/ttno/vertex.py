"""
Provides the vertex class to be used with state diagrams.
"""
from __future__ import annotations
from typing import List, Tuple
import uuid

class Vertex():
    """
    A vertex in a state diagram.

    A vertex is a point in the state diagram, which corresponds to an edge in
    the underlying tree structure. It is connected to hyperedges, which are
    correspond to the nodes of the tree structure.

    Attributes:
        corr_edge (Tuple[str, str]): The two node identifiers of the edge
            corresponding to this vertex.
        hyperedges (List[HyperEdge]): The hyperedges connected to this vertex.
        identifier (str): A unique identifier for this vertex.
        contained (bool): Whether this vertex is contained in the state
            diagram.
        new (bool): Whether this vertex is new in the state diagram.
        index (int): The index of the vertex in the TTNO bond.
    """

    def __init__(self, corr_edge: Tuple[str,str],
                 hyperedges: List[HyperEdge]):
        """
        Initialises a vertex.
        """
        self.corr_edge = corr_edge
        self.hyperedges = hyperedges

        self.identifier = str(uuid.uuid1())

        self.contained = False
        self.new = False

        # Needed for runtime-reset
        self._already_checked = False

        # Fixes the vertices value in the TTNO bond.
        # Needed to obtain an TTNO
        self.index = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the vertex.
        """
        string = f"corr_edge = {str(self.corr_edge)}; "
        string += "connected to "
        for he in self.hyperedges:
            string += f"({he.label}, {he.corr_node_id}, {he.identifier}), "

        return string

    def add_hyperedge(self, hyperedge: HyperEdge):
        """
        Adds a hyperedge to this vertex and this vertex to the hyperedge's
        vertices.
        """
        self.hyperedges.append(hyperedge)
        hyperedge.vertices.append(self)

    def add_hyperedges(self, hyperedges: List[HyperEdge]):
        """
        Adds hyperedges to this vertex and adds this vertex to the hyperedges'
            vertices.
        """
        self.hyperedges.extend(hyperedges)
        for hyperedge in hyperedges:
            hyperedge.vertices.append(self)

    def check_validity_of_node(self, node_id:str):
        """
        Cecks if the current vertex corresponds to an edge of the given node.
        """
        if not node_id in self.corr_edge:
            err_string = f"Vertex does not correspond to an edge connecting node with identifier {node_id}"
            raise ValueError(err_string)

    def get_hyperedges_for_one_node_id(self,
                                       node_id: str) -> List[HyperEdge]:
        """
        Find all hyperdeges of this vertex which correspond a given node.

        Args:
            node_id (str): The identifier of the node for which we are to find the
                hyperedges connected to this vertex.
        
        Returns:
            List[HyperEdge]: A list containing all hyperedges connected to this vertex
                and corresponding to the node with identifier ``node_id``.
        """
        self.check_validity_of_node(node_id)

        hyperedges_of_node = [hyperedge for hyperedge in self.hyperedges
                               if hyperedge.corr_node_id == node_id]

        return hyperedges_of_node

    def num_hyperedges_to_node(self, node_id: str) -> int:
        """
        Finds the number of hyperedges corresponding to a given node and are 
        connected to this vertex.

        Args:
            node_id (str): The identifier of the node for which we are to find
                the number of hyperedges connected to this vertex.
            
        Returns:
            int: The number of hyperedges of this vertex corresponding to the
                specified node.
        """
        self.check_validity_of_node(node_id)

        hyperedges = self.get_hyperedges_for_one_node_id(node_id)
        return len(hyperedges)

    def check_hyperedge_uniqueness(self, node_id: str) -> bool:
        """
        Check if the hyperedge connected to this vertex and corresponding to
        the given node is unique.

        Args:
            node_id (str): The identifier of the node for which we are to check
                the uniqueness of the hyperedge connected to this vertex.
            
        Returns:
            bool: True if the hyperedge is unique, False otherwise.
        """
        num_he = self.num_hyperedges_to_node(node_id)
        if num_he == 1:
            return True
        return False

    def get_second_node_id(self, node_id):
        """
        Given a node_id returns the other node_id of the corresponding edge.

        Args:
            node_id (string): One of the two node identifiers of the edge to
                which this vertex corresponds.

        Returns:
            string: The other node identifier.
        """
        self.check_validity_of_node(node_id)
        return [other_node_id for other_node_id in self.corr_edge
                if other_node_id != node_id][0]

    def runtime_reset(self):
        """
        Allows for reset immediately during running while building state
        diagramms.

        This works, since we check every marked vertex exactly twice.
        """
        if self._already_checked:
            self._already_checked = False
            self.contained = False
            self.new = False
        else:
            self._already_checked = True
