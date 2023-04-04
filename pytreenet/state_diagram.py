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

    def __repr__(self):
        string = "edge = " + str(self.corr_edge)
        string += "no. hyperedges = " + str(len(self.hyperedges)) + "|"
        return string
    
    def check_validity_of_node(self, node_id):
        if not (node_id in self.corr_edge):
            raise ValueError("Vertex does not correspond to and edge connecting node with identifier 'node_id'")
    
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

    def __init__(self, reference_tree):
        """
        Hyperedge collections are keyed by the node_id they correspond to,
        while vertex collections are keyed by the edge they correspond to.
        """
        self.vertex_colls = {}
        self.hyperedge_colls = {}
        
        self.reference_tree = reference_tree

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
    
    @staticmethod
    def from_hamiltonian(hamiltonian, ref_tree):

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
            hyperedge_term = temp_state_diagram.hyperedge_colls[leaf_id][0]
            hyperedge_coll = self.hyperedge_colls[leaf_id]
            potential_start_hyperedges = [he for he in hyperedge_coll
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
                    
                    unmarked_child_id = child_vertex
                    child_vertex.contained = True
                    unmarked_child_id = child_id
                    
            if unmarked_child_id == None:
                # In this case we are done due to path taken before
                print("This should never happen, but it did.")
                return
            
            new_he = self._find_new_he(child_vertex, unmarked_child_id, temp_state_diagram)
                   
        else:
            raise ValueError("Direction for an ActivePath has to be 'up' or 'down'")
        
        if new_he == None:
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
                            if hyperedge == single_term_sd.hyperedge_colls[node_id]]
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
        hyperedges = self.hyperedge_colls[node_id]
        hyperedge = [he for he in hyperedges if he.all_vertices_contained()]
        
        if len(hyperedge) == 0:
            
            
class ActivePath:
    def __init__(self, current_he, current_node_id = None, direction = "up"):
        
        self._current_he = current_he
        
        # Mainly to allow for easier access
        if current_node_id == None:
            self._current_node_id = self.current_he.corr_node_id
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
            