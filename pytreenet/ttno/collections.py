from __future__ import annotations
from typing import List, Union, Tuple

class HyperEdgeColl():
    """
    Holds multiple hyperedges that correspond to the same node.
    """

    def __init__(self, corr_node_id: str,
                 contained_hyperedges: Union[None, List[HyperEdge]] = None) -> None:
        """


        Args:
            corr_node_id (str): Identifier of a node to which this collection
             should correspond
            contained_hyperedges (nion[None, List[HyperEdge]]): A list of
             hyperedges that are already included in this collection.
             Default is None
        """
        self.corr_node_id = corr_node_id
        if contained_hyperedges is None:
            contained_hyperedges = []
        self.contained_hyperedges = contained_hyperedges

    def get_all_labels(self) -> List[str]:
        """
        Obtain all lables of all hyperedges in this collection.

        Returns:
            List[str]: A list of all labels.
        """
        return [he.label for he in self.contained_hyperedges]

    def get_hyperedges_by_label(self, label: str) -> List[HyperEdge]:
        """
        Returns all hyperedges in this collection with label 'label'.

        Args:
            label (str): The label to be looked for.

        Returns:
            List[HyperEdge]: A list of all hyperedges with the label given.
        """
        return [hyper_edge for hyper_edge in self.contained_hyperedges
                if hyper_edge.label == label]

    def get_completely_contained_hyperedges(self) -> List[HyperEdge]:
        """
        Returns all hyperedges that are connected only to vertices
        that are marked as contained.
        """
        return [hyperedge for hyperedge in self.contained_hyperedges
                if hyperedge.all_vertices_contained()]


class VertexColl:
    """
    Holds multiple vertices that correspond to the same HyperEdge.
    """

    def __init__(self, corr_edge: Tuple[str, str],
                 contained_vertices: Union[None, List[HyperEdge]]) -> None:
        """
        Contains the identifiers of two nodes which are connected by
            the edge corresponding to this collection of vertices.

        Args:
            corr_edge (Tuple[str, str]): Contains the identifiers of two nodes which
             are connected by the edge corresponding to this collection of
             vertices.
            contained_vertices (Union[None, List[HyperEdge]]): A list of
             vertices already contained in this collection.
        """
        self.corr_edge = corr_edge
        if contained_vertices is None:
            contained_vertices = []
        self.contained_vertices = contained_vertices

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
        Indexes all vertices contained in this collection. This index is the
         index value to which this vertex corresponds in the bond dimension.
        """
        for index, vertex in enumerate(self.contained_vertices):
            vertex.index = index
