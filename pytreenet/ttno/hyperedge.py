import uuid


class HyperEdge():
    def __init__(self, corr_node_id: str, label: str, vertices: list):
        self.corr_node_id = corr_node_id
        self.label = label
        self.vertices = vertices

        self.identifier = str(uuid.uuid1())

    def __repr__(self):
        string = "label = " + self.label + "; "
        string += "corr_site = " + self.corr_node_id + "; "

        string += "connected to "
        for vertex in self.vertices:
            string += str(vertex.corr_edge) + ", "
        return string

    def __eq__(self, other_he):
        labels_eq = self.label == other_he.label
        corr_node_id_eq = self.corr_node_id == other_he.corr_node_id
        return labels_eq and corr_node_id_eq
    
    def add_vertex(self, vertex):
        """
        Adds a vertex to this hyperedge and adds this hyperedge to the vertex's hyperedges.
        """
        self.vertices.append(vertex)
        vertex.hyperedges.append(self)

    def add_vertices(self, vertices: list):
        """
        Adds vertices to this hyperedge and adds this hyperedge to the vertices' hyperedges.
        """
        self.vertices.extend(vertices)
        for vertex in vertices:
            vertex.hyperedges.append(self)

    def find_vertex(self, other_node_id):
        """
        Finds the vertex connected to this hyperedge and corresponds to the edge
        (corr_node_id, other_node_id).

        Parameters
        ----------
        other_node_id : string
            The vertex corresponds to an edge in the underlying tree structure.
            One node the edge is connected to is the node this hyperedge
            corresponds to, while the other node identifier is provided by this
            string

        Returns
        -------
        vertex: Vertex
            The vertex corresponding to the edge connecting self.corr_node
            and other_node_id
        """
        vertex_list = [vertex for vertex in self.vertices
                       if other_node_id in vertex.corr_edge]

        if len(vertex_list) == 0:
            err_str = f"Hyperedge not connected to a vertex corresponding to edge {(self.corr_node_id, other_node_id)}!"
            raise ValueError(err_str)
        assert len(vertex_list) == 1, "Hyperedge should only be connected to one vertex corresponding to an edge!"

        return vertex_list[0]

    def vertex_single_he(self, other_node_id):
        """
        Checks if the vertex with the corresponding edge (corr_node_id, other_node_id)
        is connected to a single hyperedge at the current node.

        Parameters
        ----------
        other_node_id : string
            The vertex corresponds to an edge in the underlying tree structure.
            One node the edge is connected to is the node this hyperedge
            corresponds to, while the other node identifier is provided by this
            string

        Returns
        -------
        result: bool
        """
        vertex = self.find_vertex(other_node_id)

        return vertex.check_hyperedge_uniqueness(self.corr_node_id)

    def get_contained_vertices(self):
        """
        Returns all vertices marked as contained connected to this hyperedge.
        """
        return [vertex for vertex in self.vertices if vertex.contained]

    def get_uncontained_vertices(self):
        """
        Returns all vertices not marked as contained connected to this hyperedge.
        """
        return [vertex for vertex in self.vertices if not vertex.contained]

    def get_single_uncontained_vertex(self):
        """
        Returns a single uncontained vertex. Throws an error, if there are more or less than one.
        """
        uncontained_vertices = self.get_uncontained_vertices()
        err_string = "The hyperedge is not connected to a single contained vertex."
        assert len(uncontained_vertices) == 1, err_string
        return uncontained_vertices[0]

    def num_of_vertices_contained(self):
        """
        Determines the number of vertices connected to this hyperedge
        which are marked as contained.

        Returns
        -------
        num_marked_contained: int

        """
        marked_contained = self.get_contained_vertices()
        num_marked_contained = len(marked_contained)
        return num_marked_contained

    def all_vertices_contained(self):
        """
        Checks, if all vertices attached to this hyperedge are marked as
        contained.
        """
        return len(self.get_uncontained_vertices()) == 0

    def all_but_one_vertex_contained(self):
        """
        Checks, if all but one vertex of attached to this hyperedge are
        marked as contained.

        Returns
        -------
        result: bool

        """
        return len(self.get_uncontained_vertices()) == 1
