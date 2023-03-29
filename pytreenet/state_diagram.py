from .util import compare_lists_by_value

class HyperEdge():
    def __init__(self, corr_node_id, label, vertices):
        self.corr_node_id = corr_node_id
        self.label = label
        self.vertices = vertices

    def __repr__(self):
        string = "node_id = " + self.corr_node_id + ";"
        string += "label = " + self.label + ";"
        string += "no. vertices = " + str(len(self.vertices)) + "|"
        return string

    def __eq__(self, other_he):
        labels_eq = self.label == other_he.label
        corr_node_id_eq = self.corr_node_id
        return labels_eq and corr_node_id_eq

class Vertex():
    def __init__(self, corr_edge, hyperedges):
        self.corr_edge = corr_edge
        self.hyperedges = hyperedges

    def __repr__(self):
        string = "edge = " + str(self.corr_edge)
        string += "no. hyperedges = " + str(len(self.hyperedges)) + "|"
        return string

class HyperEdgeColl():
    def __init__(self, corr_node_id, contained_hyperedges):
        self.corr_node_id = corr_node_id
        self.contained_hyperedges = contained_hyperedges

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

    def __repr__(self):
        string = str(self.corr_edge) + "\n"
        string += str(self.contained_vertices)
        return string

class StateDiagram():

    def __init__(self):
        self.vertex_colls = {}
        self.hyperedge_colls = {}

    def __repr__(self):
        string = str(self.vertex_colls)
        string += str(self.hyperedge_colls)
        return string

    def get_all_vertices(self):
        all_vert = []
        for vertex_coll in self.vertex_colls:
            all_vert.extend(vertex_coll.contained_vertices)
        return all_vert

    @staticmethod
    def from_single_term(term, reference_tree):
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

        sd = StateDiagram()
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

    @staticmethod
    def from_hamiltonian(hamiltonian, ref_tree):

        all_nodes_list = ref_tree.nodes.keys()
        state_diagram = None

        for term in hamiltonian.terms:
            if state_diagram == None:
                state_diagram = StateDiagram.from_single_term(term, ref_tree)
            else:
                temp_state_diagram = StateDiagram.from_single_term(term, ref_tree)
                not_visited = temp_state_diagram.get_all_vertices()

                while len(not_visited) > 0:
                    current_vertex = not_visited[0]
                    path_origins = temp_state_diagram.find_path_with_origin(current_vertex)
                    for path_origin in path_origins:
                        state_diagram.contains_path(path_origin)

    def contains_path(self, path_origin):
        """
        Checks, if path is contained in the current state diagram

        Parameters
        ----------
        path : list
            A path with a certain origin.

        Returns
        -------
        None.

        """
        contains = True

        origin = path_origin.now
        target_hyperedges = path_origin.targets
        next_node_id = target_hyperedges[0].corr_node_id # The node_id corresponding to the next step in the path

        # For all vertices already in the diagram at the same edge, we check if they are connected to equivalent hyperedges
        for vertex in self.vertex_colls[str(origin.corr_edge)]:
            hyperedges_of_diagram_vertex = [hyperedge for hyperedge in vertex.hyperedges
                                 if hyperedge.corr_node_id == next_node_id]

            if compare_lists_by_value(target_hyperedges, hyperedges_of_diagram_vertex):



    def find_path_with_origin(self, origin):
        """


        Parameters
        ----------
        origin : Vertex
            The origin vertex, of the path.

        Returns
        -------
        origin_path_element1, origin_path_element2: PathElements
        The oigin of the path. They have a target attribute pointing to the
        next path element.

        """

        node1_id = origin.corr_edge[0]
        origin_path_element1 = PathElement(origin, [])
        self._add_hyperedges_to_path(origin_path_element1, node1_id)

        node2_id = origin.corr_edge[1]
        origin_path_element2 = PathElement(origin, [])
        self._add_hyperedges_to_path(origin_path_element2, node2_id)

        return origin_path_element1, origin_path_element2

    def _add_hyperedges_to_path(self, source_path_element, node_id_as_direction):
        hyperedges_to_add = [hyperedge for hyperedge in source_path_element.now.hyperedges
                             if hyperedge.corr_node_id == node_id_as_direction]

        for hyperedge in hyperedges_to_add:
            he_path_element = PathElement(hyperedge, [])
            source_path_element.targets.append(he_path_element)
            self._add_vertices_to_path(he_path_element, source_path_element.now)

    def _add_vertices_to_path(self, source_path_element, last_vertex):
        vertices_to_add = [vertex for vertex in source_path_element.now.vertices
                           if vertex != last_vertex]

        current_node_id = source_path_element.now.corr_node_id
        for vertex in vertices_to_add:
            node_id_as_direction = [node_id for node_id in vertex.corr_edge
                                    if node_id != current_node_id][0]
            vertex_path_element = PathElement(vertex, [])
            source_path_element.targets.append(vertex_path_element)

            self._add_hyperedges_to_path(vertex_path_element, node_id_as_direction)

class PathElement:
    def __init__(self, now, targets):
        self.now = now
        self.targets = targets

    def __repr__(self):
        string = str(self.now) + " & "
        string += str(len(self.targets))
        return string