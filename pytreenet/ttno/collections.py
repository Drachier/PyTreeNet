class HyperEdgeColl():
    """
    Holds multiple hyperedges, that correspond to the same node.
    """
    def __init__(self, corr_node_id, contained_hyperedges: list):
        self.corr_node_id = corr_node_id
        self.contained_hyperedges = contained_hyperedges

    def get_all_labels(self):
        """
        Obtain all lables of all hyperedges in this collection.
        """
        return [he.label for he in self.contained_hyperedges]

    def get_hyperedges_by_label(self, label):
        """
        Returns all hyperedges in this collection with label 'label'.
        """
        return [hyper_edge for hyper_edge in self.contained_hyperedges
                if hyper_edge.label == label]

    def get_completely_contained_hyperedges(self):
        """
        Returns all hyperedges that are connected only to vertices
        that are marked as contained.
        """
        return [hyperedge for hyperedge in self.contained_hyperedges
                if hyperedge.all_vertices_contained()]

class VertexColl():
    def __init__(self, corr_edge: tuple, contained_vertices: list):
        """
        Parameters
        ----------
        corr_edge : tuple of str
            Contains the identifiers of the two nodes, which are connected by
            the edge corresponding to this collection of vertices.
        contained_vertices : list of Vertex

        Returns
        -------
        None.

        """
        self.corr_edge = corr_edge
        self.contained_vertices = contained_vertices

    def contains_contained(self):
        """
        Returns, if one of the vertices in this collection are marked as
        contained.

        Returns
        -------
        result: bool

        """
        all_marked_vertices = [vertex for vertex in self.contained_vertices
                               if vertex.contained]
        if len(all_marked_vertices) == 0:
            return False
        return True

    def get_all_marked_vertices(self):
        """
        Returns all vertices in this collection marked as contained or new.
        """
        return [vertex for vertex in self.contained_vertices
                if vertex.contained or vertex.new]