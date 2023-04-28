from copy import copy

class HyperEdge():
    def __init__(self, corr_node_id, label, vertices):
        self.corr_node_id = corr_node_id
        self.label = label
        self.vertices = vertices

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

    def find_correct_vertex(self, other_node_id):
        """

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
            The vertex corresponding to the edge connecting self.corr_node#
            and other_node_id
        """
        vertex_list = [vertex for vertex in self.vertices if
                           other_node_id in vertex.corr_edge]
        assert len(vertex_list) == 1
        return vertex_list[0]

    def vertex_single_he(self, other_node_id):
        """
        Checks if the vertex with the corresponding edge vertex_corr_edge
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
        vertex = self.find_correct_vertex(other_node_id)

        return vertex.check_hyperedge_uniqueness(self.corr_node_id)

    def num_of_vertices_contained(self):
        """
        Determines the number of vertices connected to this hyperedge
        which are marked as contained.

        Returns
        -------
        num_marked_contained: int

        """
        marked_contained = [True for vertex in self.vertices
                                if vertex.contained]
        num_marked_contained = len(marked_contained)
        return num_marked_contained

    def all_vertices_contained(self):
        """
        Checks, if all vertices attached to this hyperedge are marked as
        contained.
        """
        num_vertices = len(self.vertices)
        if num_vertices == self.num_of_vertices_contained():
            return True
        return False

    def all_but_one_vertex_contained(self):
        """
        Checks, if all or all but one vertex of attached to this hyperedge are
        marked as contained.

        Returns
        -------
        result: bool

        """
        num_vertices = len(self.vertices)
        num_marked_contained = self.num_of_vertices_contained()

        assert num_vertices >= num_marked_contained
        if num_vertices == num_marked_contained:
            return True
        elif num_vertices == num_marked_contained + 1:
            return True

        return False

class Vertex():
    def __init__(self, corr_edge, hyperedges):
        self.corr_edge = corr_edge
        self.hyperedges = hyperedges

        self.contained = False
        self.new = False

        # Needed to obtain an MPO
        self.index_value = None

    def __repr__(self):
        string = "corr_edge = " + str(self.corr_edge) + "; "
        string += "connected to "
        for he in self.hyperedges:
            string += "(" + he.label + ", " + he.corr_node_id + "), "

        return string

    def check_validity_of_node(self, node_id):
        if not (node_id in self.corr_edge):
            raise ValueError(f"Vertex does not correspond to an edge connecting node with identifier {node_id}")

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

        assert len(hyperedges_of_node) > 0

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

        he = self.get_hyperedges_for_one_node_id(node_id)
        return len(he)

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

class HyperEdgeColl():
    def __init__(self, corr_node_id, contained_hyperedges):
        self.corr_node_id = corr_node_id
        self.contained_hyperedges = contained_hyperedges

    def get_all_labels(self):
        """
        Obtain all lables of all hyperedges in this collection.
        """
        return [he.label for he in self.contained_hyperedges]

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

        # Required to obtain the TTNO later on
        self.leg_index = None

class StateDiagram():

    def __init__(self, reference_tree):
        """
        Hyperedge collections are keyed by the node_id they correspond to,
        while vertex collections are keyed by the edge they correspond to.
        """
        self.vertex_colls = {}
        self.hyperedge_colls = {}

        self.reference_tree = reference_tree

    def __repr__(self):
        all_he = self.get_all_hyperedges()
        all_vert = self.get_all_vertices()

        string = "hyperedges:\n"
        for he in all_he:
            string += str(he) + "\n"
        string += "\n vertices:\n"
        for vertex in all_vert:
            string += str(vertex) + "\n"

        return string

    def get_all_vertices(self):
        all_vert = []
        for vertex_coll_id in self.vertex_colls:
            all_vert.extend(self.vertex_colls[vertex_coll_id].contained_vertices)
        return all_vert

    def get_all_hyperedges(self):
        all_he = []
        for he_coll_id in self.hyperedge_colls:
            all_he.extend(self.hyperedge_colls[he_coll_id].contained_hyperedges)
        return all_he

    def get_vertex_coll_two_ids(self, id1, id2):
        """
        Obtain the vertex collection that corresponds to the edge between nodes
        with identifiers id1 and id2. Since the order of the identifiers could
        be either way, we have to check both options.

        Parameters
        ----------
        id1, id2: str
            Identifiers of two nodes in the reference tree.

        Returns
        -------
        vertex_coll: VertexColl
            The vertex collection corresponding to the edge connecting the nodes
            with identifers id1 and id2.

        """
        key1 = (id1, id2)
        key2 = (id2, id1)
        if key1 in self.vertex_colls:
            return self.vertex_colls[key1]
        elif key2 in self.vertex_colls:
            return self.vertex_colls[key2]
        else:
            raise KeyError(f"There is no vertex collection corresponding to and edge between {id1} and {id2}")

    def add_hyperedge(self, hyperedge):
        """
        Adds a hyperedge to the correct hyperedge collection and thus to the
        state diagram.

        Parameters
        ----------
        hyperedge : HyperEdge
            Hyperedge to be added to the diagram.

        Returns
        -------
        None.

        """
        node_id = hyperedge.corr_node_id

        if node_id in self.reference_tree.nodes:
            self.hyperedge_colls[node_id].contained_hyperedges.append(hyperedge)
        else:
            raise KeyError(f"No node with identifier {node_id} in reference tree.")

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

        sd = StateDiagram(reference_tree)
        sd._from_single_term_rec(VertexColl((None,reference_tree.root_id), [None]), term, reference_tree)

        return sd

    def _from_single_term_rec(self, vertex_coll, term, reference_tree):
        new_node_id = vertex_coll.corr_edge[1]
        old_vertex = vertex_coll.contained_vertices[0]
        node = reference_tree.nodes[new_node_id]

        new_hyperedge = HyperEdge(new_node_id, term[new_node_id], [old_vertex])
        if not node.is_root():
            old_vertex.hyperedges.append(new_hyperedge)

        new_hyperedge_coll = HyperEdgeColl(new_node_id, [new_hyperedge])

        if node.is_leaf():
            self.hyperedge_colls[new_node_id] = new_hyperedge_coll
            return

        vertices = []
        vertex_colls = []
        for child_id in node.children_legs:
            vertex = Vertex((new_node_id, child_id),[new_hyperedge])
            vertices.append(vertex)
            new_vertex_coll = VertexColl((new_node_id, child_id), [vertex])
            vertex_colls.append(new_vertex_coll) # Required for the recursion
            self.vertex_colls[(new_node_id, child_id)] = new_vertex_coll

        if node.is_root():
            new_hyperedge.vertices = vertices
        else:
            new_hyperedge.vertices.extend(vertices)
        self.hyperedge_colls[new_node_id] = HyperEdgeColl(new_node_id, [new_hyperedge])

        for new_vertex_coll in vertex_colls:
            self._from_single_term_rec(new_vertex_coll, term, reference_tree)

    @classmethod
    def from_hamiltonian(cls, hamiltonian, ref_tree):

        state_diagram = None

        for term in hamiltonian.terms:
            if state_diagram == None:
                state_diagram = StateDiagram.from_single_term(term, ref_tree)
            else:
                state_diagram.add_single_term(term)

        return state_diagram

    def add_single_term(self, term):
        temp_state_diagram = StateDiagram.from_single_term(term, self.reference_tree)

        leaves = self.reference_tree.get_leaves()
        for leaf_id in leaves:
            hyperedge_term = temp_state_diagram.hyperedge_colls[leaf_id].contained_hyperedges[0]
            hyperedge_coll = self.hyperedge_colls[leaf_id]
            potential_start_hyperedges = [he for he in hyperedge_coll.contained_hyperedges
                                              if he == hyperedge_term]

            parent_id = self.reference_tree.nodes[leaf_id].parent_leg[0]

            # All hyperedges whose parent vertex has a single child hyperedge
            single_hyperedges = [he for he in potential_start_hyperedges
                                     if he.vertex_single_he(parent_id)]

            # Should be max a single element, otherwise something went wrong before
            assert len(single_hyperedges) == 0 or len(single_hyperedges) == 1

            if len(single_hyperedges) == 1:
                active_path = ActivePath(single_hyperedges[0])
                self.run_active_path(active_path, temp_state_diagram)

        self._add_hyperedges(temp_state_diagram)
        self.reset_markers()

    def run_active_path(self, active_path, temp_state_diagram):

        if active_path.current_node_id == self.reference_tree.root_id:
            active_path.direction = "down"

        if active_path.direction == "up":
            parent_id = self.reference_tree.nodes[active_path.current_node_id].parent_leg[0]
            parent_vertex = active_path.current_he.find_correct_vertex(parent_id)

            if parent_vertex.contained:
                active_path.direction = "down"
                self.run_active_path(active_path, temp_state_diagram)
                return

            if not parent_vertex.check_hyperedge_uniqueness(active_path.current_node_id):
                return

            parent_vertex.contained = True
            new_he = self._find_new_he(parent_vertex, parent_id, temp_state_diagram)

        elif active_path.direction == "down":
            current_node = self.reference_tree.nodes[active_path.current_node_id]

            for child_id in current_node.children_legs:
                child_vertex = active_path.current_he.find_correct_vertex(child_id)
                # Otherwise multiple paths would be created
                if not child_vertex.check_hyperedge_uniqueness(active_path.current_node_id):
                    return

            count = 0
            unmarked_child_id = None
            for child_id in current_node.children_legs:
                child_vertex = active_path.current_he.find_correct_vertex(child_id)

                if not child_vertex.contained:
                    # There should only be a single unmarked child_index
                    assert count == 0
                    count += 1

                    unmarked_child_vertex = child_vertex
                    child_vertex.contained = True
                    unmarked_child_id = child_id

            if unmarked_child_id == None:
                # In this case we are done due to path taken before
                print("This should never happen, but it did.")
                return

            new_he = self._find_new_he(unmarked_child_vertex, unmarked_child_id, temp_state_diagram)

        else:
            raise ValueError("Direction for an ActivePath has to be 'up' or 'down'")

        if type(new_he) == type(None):
            return
        else:
            # In this case there is a he contained in the current state diagram
            # that corresponds to the same he in the single term sd
            # Thus we continue the current path
            active_path.current_he = new_he
            self.run_active_path(active_path, temp_state_diagram)

    def _find_new_he(self, current_vertex, node_id, single_term_sd):
        """
        If there is a new hyperedge in the current statediagram, which
        is equivalent to the hyperedge in the single term StateDiagram, return it.
        Else return None

        Parameters
        ----------
        current_vertex : Vertex
            The vertex from which we are currently looking for potential
            hyperedges.
        node_id : str
            The identifier of the node to which the potential hyperedge
            corresponds.

        Returns
        -------

        """
        he_of_vertex = current_vertex.get_hyperedges_for_one_node_id(node_id)

        potential_new_he = [hyperedge for hyperedge in he_of_vertex
                                if hyperedge.all_but_one_vertex_contained()]

        potential_new_he = [hyperedge for hyperedge in potential_new_he
                            if hyperedge == single_term_sd.hyperedge_colls[node_id].contained_hyperedges[0]]
        # All checks done

        assert len(potential_new_he) <= 1

        if len(potential_new_he) == 1:
            new_he = potential_new_he[0]
            return new_he

        return None

    def _add_hyperedges(self, temp_state_diagram):

        root_id = self.reference_tree.root_id

        self._add_hyperedges_rec(root_id, temp_state_diagram)

    def _add_hyperedges_rec(self, node_id, temp_state_diagram):
        hyperedge_coll = self.hyperedge_colls[node_id]
        hyperedge = [he for he in hyperedge_coll.contained_hyperedges if
                     he.all_vertices_contained()]

        node = self.reference_tree.nodes[node_id]
        neighbour_ids = node.neighbouring_nodes(with_legs=False)

        potential_label = temp_state_diagram.hyperedge_colls[node_id].contained_hyperedges[0].label

        if len(hyperedge) == 0:
            vertices_to_connect_to_new_he = []

            for neighbour_id in neighbour_ids:
                vertex_coll = self.get_vertex_coll_two_ids(node_id, neighbour_id)
                vertex_to_connect = [vertex for vertex in vertex_coll.contained_vertices
                                     if (vertex.contained or vertex.new)]
                # TODO: Vertices are checked a maximum of two times, so the markers could be reset after doing this twice
                if len(vertex_to_connect) == 0:
                    vertex_to_connect = Vertex((node_id, neighbour_id), [])
                    vertex_to_connect.new = True
                    vertex_coll.contained_vertices.append(vertex_to_connect)
                elif len(vertex_to_connect) == 1:
                    vertex_to_connect = vertex_to_connect[0]
                else:
                    assert False, "Something went terribly wrong!"

                vertices_to_connect_to_new_he.append(vertex_to_connect)

            new_hyperedge = HyperEdge(node_id, potential_label, vertices_to_connect_to_new_he)
            self.add_hyperedge(new_hyperedge)

            for vertex in vertices_to_connect_to_new_he:
                vertex.hyperedges.append(new_hyperedge)

        elif len(hyperedge) == 1 and (hyperedge[0].label != potential_label):
            vertices_to_connect_to_new_he = copy(hyperedge[0].vertices)
            new_hyperedge = HyperEdge(node_id, potential_label, vertices_to_connect_to_new_he)
            self.add_hyperedge(new_hyperedge)

            for vertex in vertices_to_connect_to_new_he:
                vertex.hyperedges.append(new_hyperedge)

        elif len(hyperedge) > 1:
            assert False, "Something went terribly wrong!"

        children_ids = node.get_children_ids()

        for child_id in children_ids:
            self._add_hyperedges_rec(child_id, temp_state_diagram)

    def reset_markers(self):
        """
        Resets the contained and new markers of every vertex in the state diagram.

        Returns
        -------
        None.

        """
        for vertex_col_corr in self.vertex_colls:
            vertex_col = self.vertex_colls[vertex_col_corr]
            for vertex in vertex_col.contained_vertices:
                vertex.contained = False
                vertex.new = False

class ActivePath(object):
    def __init__(self, current_he, current_node_id = None, direction = "up"):

        self._current_he = current_he

        # Mainly to allow for easier access
        if current_node_id == None:
            self._current_node_id = self._current_he.corr_node_id
        else:
            self._current_node_id = current_node_id

        if direction == "up" or direction == "down":
            self.direction = direction
        else:
            raise ValueError("Direction for an ActivePath has to be 'up' or 'down'")

    @property
    def current_he(self):
        return self._current_he

    @current_he.setter
    def current_he(self, new_he):
        self._current_he = new_he
        self._current_node_id = new_he.corr_node_id

    @property
    def current_node_id(self):
        return self._current_node_id
