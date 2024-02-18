import uuid

class Vertex():
    def __init__(self, corr_edge: tuple, hyperedges: list):
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

    def __repr__(self):
        string = "corr_edge = " + str(self.corr_edge) + "; "
        string += "connected to "
        for he in self.hyperedges:
            string += "(" + he.label + ", " + he.corr_node_id + "), "

        return string

    def add_hyperedge(self, hyperedge):
        """
        Adds a hyperedge to this vertex and adds this vertex to the hyperedge's vertices.
        """
        self.hyperedges.append(hyperedge)
        hyperedge.vertices.append(self)

    def add_hyperedges(self, hyperedges: list):
        """
        Adds hyperedges to this vertex and adds this vertex to the hyperedges' vertices.
        """
        self.hyperedges.extend(hyperedges)
        for hyperedge in hyperedges:
            hyperedge.vertices.append(self)

    def check_validity_of_node(self, node_id):
        """Cecks if the current vertex corresponds to an edge of node_id."""
        if not node_id in self.corr_edge:
            err_string = f"Vertex does not correspond to an edge connecting node with identifier {node_id}"
            raise ValueError(err_string)

    def get_hyperedges_for_one_node_id(self, node_id):
        """
        Find all hyperedges connected to this vertex which correspond to the node with
        identifier 'node_id'

        node_id : str
            The identifier of the node for which we are to find the
            hyperedges connected to this vertex

        Returns
        -------
        hyperedges_of_node: list of HyperEdge
            A list containing all hyperedges connected to this vertex and
            corresponding to the node with identifier node_id.
        """
        self.check_validity_of_node(node_id)

        hyperedges_of_node = [hyperedge for hyperedge in self.hyperedges
                               if hyperedge.corr_node_id == node_id]

        return hyperedges_of_node

    def num_hyperedges_to_node(self, node_id):
        """
        Finds the number of hyperedges which are connected to this
        vertex and correspond to the node with identifier 'node_id'.

        Parameters
        ----------
        node_id : str
            Identifier of a node that is connected by the edge
            corresponding to this vertex.

        Returns
        -------
        num_he: int
            The number of hyperedges of this vertex corresponding
            to the node with identifer 'node_id'.

        """
        self.check_validity_of_node(node_id)

        hyperedges = self.get_hyperedges_for_one_node_id(node_id)
        return len(hyperedges)

    def check_hyperedge_uniqueness(self, node_id):
        """
        Checks, if the number of hyperedges attached to this vertex and
        corresponding to the node with identifier 'node_id' is one.
        """
        num_he = self.num_hyperedges_to_node(node_id)

        if num_he == 1:
            return True
        else:
            return False

    def get_second_node_id(self, node_id):
        """Given a node_id returns the other node_id of the corresponding edge

        Args:
            node_id (string): One of the two node_ids of the edge to which this vertex corresponds.

        Returns:
            string: The other node_id.
        """
        self.check_validity_of_node(node_id)
        return [other_node_id for other_node_id in self.corr_edge
                if other_node_id != node_id][0]

    def runtime_reset(self):
        """
        Allows for reset immediately during running while building state diagramms.

        This works, since we check every marked vertex exactly twice.
        """
        if self._already_checked:
            self._already_checked = False
            self.contained = False
            self.new = False
        else:
            self._already_checked = True
