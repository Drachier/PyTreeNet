from __future__ import annotations
from typing import Dict, Tuple
from copy import copy
from collections import deque

from enum import Enum
from .vertex import Vertex
from .hyperedge import HyperEdge
from .collections import VertexColl, HyperEdgeColl
from .single_term_diagram import SingleTermDiagram
from ..bipartite_graph import BipartiteGraph, minimum_vertex_cover


class method(Enum):
    TREE = "Tree"
    BIPARTITE = "Bipartite"

class StateDiagram():
    """ 
    A state diagram representing a Hamiltonian.
    Contains collections of vertices and hyperedges as
    well as a reference tree.
    """

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
        for hyperedge in all_he:
            string += str(hyperedge) + "\n"
        string += "\n vertices:\n"
        for vertex in all_vert:
            string += str(vertex) + "\n"

        return string

    def get_all_vertices(self):
        """
        Returns all vertices from all collections in a list.
        """
        all_vert = []
        for vertex_coll in self.vertex_colls.values():
            all_vert.extend(vertex_coll.contained_vertices)
        return all_vert

    def get_all_hyperedges(self):
        """
        Returns all hyperedges from all collections in a list
        """
        all_he = []
        for hyperedge_coll in self.hyperedge_colls.values():
            all_he.extend(hyperedge_coll.contained_hyperedges)
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
        if key2 in self.vertex_colls:
            return self.vertex_colls[key2]
        raise KeyError(
            f"There is no vertex collection corresponding to and edge between {id1} and {id2}")

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
            self.hyperedge_colls[node_id].contained_hyperedges.append(
                hyperedge)
        else:
            raise KeyError(
                f"No node with identifier {node_id} in reference tree.")

    @classmethod
    def from_hamiltonian(cls, hamiltonian, ref_tree, method: method = method.TREE) -> StateDiagram:
        """Creates a state diagram equivalent to a given Hamiltonian

        Args:
            hamiltonian (Hamiltonian): Hamiltonian for which the state
                diagram is to be found
            ref_tree (TreeTensorNetwork): Supplies the tree topology which
                is to be incorporated into the state diagram.

        Returns:
            StateDiagram: The final state diagram
        """

        state_diagram = None

        if method == method.BIPARTITE:
            state_diagram = cls.from_hamiltonian_bipartite(hamiltonian, ref_tree)
        
        elif method == method.TREE:
            for term in hamiltonian.terms:
                if state_diagram is None:
                    state_diagram = cls.from_single_term(term, ref_tree)
                else:
                    state_diagram.add_single_term(term)

        return state_diagram

    @classmethod
    def from_single_term(cls, term, reference_tree):
        """
        Basically a wrap of 'SingleTermDiagram.from_single_term'.
        """
        single_term_diagram = SingleTermDiagram.from_single_term(
            term, reference_tree)
        return cls.from_single_state_diagram(single_term_diagram)

    @classmethod
    def from_single_state_diagram(cls, single_term_diag):
        """Transforms a single state diagram to a general one.

        Args:
            single_term_diag (SingleTermDiagram): Represents a single term
                using a simpler structure than the general state diagrams.

        Returns:
            state_diagram (StateDiagram): The equivalent general state diagram.
        """
        state_diagram = cls(single_term_diag.reference_tree)

        # Creating HyperEdgeCollections
        for node_id, hyperedge in single_term_diag.hyperedges.items():
            new_hyperedge_coll = HyperEdgeColl(node_id, [hyperedge])
            state_diagram.hyperedge_colls[node_id] = new_hyperedge_coll

        # Creating VertexCollections
        for edge_id, vertex in single_term_diag.vertices.items():
            new_vertex_coll = VertexColl(edge_id, [vertex])
            state_diagram.vertex_colls[edge_id] = new_vertex_coll

        return state_diagram

    def add_single_term(self, term: dict):
        """Modifies the state diagram to add a term.

        Adds a term to the state diagram. This means the diagram is
        modified in a way such that it represents Hamiltonian + term
        instead of only Hamiltonian.

        Args:
            term (dict): A dictionary containing the node_ids as keys
                and the operator applied to that node as a value.
        """
        single_term_diagram = SingleTermDiagram.from_single_term(
            term, self.reference_tree)

        self._mark_contained_vertices(single_term_diagram)
        self._add_hyperedges(single_term_diagram)
        # At this point all vertices have their marking reset already.

    def _mark_contained_vertices(self, single_term_diagram):
        leaves = self.reference_tree.get_leaves()
        for leaf_id in leaves:
            hyperedge_label = single_term_diagram.get_hyperedge_label(leaf_id)
            hyperedge_coll = self.hyperedge_colls[leaf_id]
            # Ensure the hyperedges have the correct label
            potential_start_hyperedges = hyperedge_coll.get_hyperedges_by_label(
                hyperedge_label)
            potential_start_hyperedges = [hyperedge for hyperedge in potential_start_hyperedges
                                          if hyperedge.all_but_one_vertex_contained()]
            next_vertex = None
            for hyperedge in potential_start_hyperedges:
                next_vertex = self._find_and_mark_new_vertex(hyperedge)
                if not next_vertex is None:
                    break

            if not next_vertex is None:
                new_node_id = next_vertex.get_second_node_id(leaf_id)
                self._find_new_he(next_vertex, new_node_id, single_term_diagram)

    def _find_and_mark_new_vertex(self, current_hyperedge):
        current_node_id = current_hyperedge.corr_node_id
        uncontained_vertex = current_hyperedge.get_single_uncontained_vertex()
        next_node_id = uncontained_vertex.get_second_node_id(current_node_id)
        if not current_hyperedge.vertex_single_he(next_node_id):
            # In this case we would add multiple paths
            return None
        vertex_coll = self.get_vertex_coll_two_ids(current_node_id, next_node_id)
        if vertex_coll.contains_contained():
            # This vertex collection already has a contained vertex
            return None
        uncontained_vertex.contained = True
        return uncontained_vertex

    def _find_new_he(self, current_vertex, node_id, single_term_diagram):

        he_of_vertex = current_vertex.get_hyperedges_for_one_node_id(node_id)
        potential_new_he = [hyperedge for hyperedge in he_of_vertex
                            if hyperedge.all_but_one_vertex_contained()]

        desired_label = single_term_diagram.get_hyperedge_label(node_id)
        potential_new_he = [hyperedge for hyperedge in potential_new_he
                            if hyperedge.label == desired_label]

        next_vertex = None
        for hyperedge in potential_new_he:
            next_vertex = self._find_and_mark_new_vertex(hyperedge)
            if not next_vertex is None:
                break

        if not next_vertex is None:
            new_node_id = next_vertex.get_second_node_id(node_id)
            self._find_new_he(next_vertex, new_node_id, single_term_diagram)

    def _add_hyperedges(self, single_term_diagram):
        for node_id in self.reference_tree.nodes:
            self._add_hyperedges_rec(node_id, single_term_diagram)

    def _add_hyperedges_rec(self, node_id, single_term_diagram):
        hyperedge_coll = self.hyperedge_colls[node_id]
        hyperedges = hyperedge_coll.get_completely_contained_hyperedges()
        desired_label = single_term_diagram.get_hyperedge_label(node_id)

        if len(hyperedges) == 0:
            vertices_to_connect_to_new_he = self._find_vertices_connecting_to_he(
                node_id)
            new_hyperedge = HyperEdge(
                node_id, desired_label, vertices_to_connect_to_new_he)

        elif len(hyperedges) >= 1:
            for hyperedge in hyperedges:
                vertices_to_connect_to_new_he = copy(hyperedge.vertices)
                if hyperedge.label == desired_label:
                    new_hyperedge = None
                    break
                else:
                    new_hyperedge = HyperEdge(
                        node_id, desired_label, vertices_to_connect_to_new_he)

        # Allows for reset directly instead of after the fact
        # Thus we don't have to run through all vertices of the entire diagram.
        for vertex in vertices_to_connect_to_new_he:
            vertex.runtime_reset()

        if not new_hyperedge is None:
            self.add_hyperedge(new_hyperedge)
            for vertex in vertices_to_connect_to_new_he:
                vertex.hyperedges.append(new_hyperedge)

    def _find_vertices_connecting_to_he(self, node_id):
        node = self.reference_tree.nodes[node_id]
        neighbour_ids = node.neighbouring_nodes()

        vertices_to_connect_to_new_he = []
        for neighbour_id in neighbour_ids:
            vertex_coll = self.get_vertex_coll_two_ids(node_id, neighbour_id)
            vertex_to_connect = vertex_coll.get_all_marked_vertices()
            if len(vertex_to_connect) == 0:
                vertex_to_connect = Vertex((node_id, neighbour_id), [])
                vertex_to_connect.new = True
                vertex_coll.contained_vertices.append(vertex_to_connect)
            elif len(vertex_to_connect) == 1:
                vertex_to_connect = vertex_to_connect[0]
            else:
                assert False, "Something went terribly wrong!"

            vertices_to_connect_to_new_he.append(vertex_to_connect)
        return vertices_to_connect_to_new_he

    def obtain_tensor_shape(self, node_id: str,
                            conversion_dict: Dict[str, np.ndarray]) -> Tuple[int, ...]:
        """
        Find the required shape of the tensor corresponding to a node in the
         equivalent TTNO.

        Args:
            node_id (str): The identifier of a node.
            conversion_dict (Dict[str, np.ndarray]): A dictionary to convert
             the labels into arrays, to determine the required physical
             dimension.

        Returns:
            Tuple[int, ...]: The shape of the tensor in the equivalent TTNO in the
             format (parent_shape, children_shape, phys_dim, phys_dim).
             The children are in the same order as in the node.
        """
        he = self.hyperedge_colls[node_id].contained_hyperedges[0]
        operator_label = he.label
        operator = conversion_dict[operator_label]
        # Should be square operators
        assert operator.shape[0] == operator.shape[1]
        phys_dim = operator.shape[0]
        total_shape = [0] * len(he.vertices)
        total_shape.extend([phys_dim, phys_dim])
        neighbours = self.reference_tree.nodes[node_id].neighbouring_nodes()
        for leg_index, neighbour_id in enumerate(neighbours):
            vertex_coll = self.get_vertex_coll_two_ids(node_id, neighbour_id)
            # The number of vertices is equal to the number of bond-dimensions
            # required.
            total_shape[leg_index] = len(vertex_coll.contained_vertices)
        return tuple(total_shape)

    def set_all_vertex_indices(self):
        """
        Indexes all vertices contained in this state diagram. This index is
         the index value to which this vertex corresponds in the bond
         dimension.
        """
        for vertex_coll in self.vertex_colls.values():
            vertex_coll.index_vertices()

    def reset_markers(self):
        """
        Resets the contained and new markers of every vertex in the state diagram.

        Returns
        -------
        None.

        """
        for vertex_col in self.vertex_colls.values():
            for vertex in vertex_col.contained_vertices:
                vertex.contained = False
                vertex.new = False

    @classmethod
    def from_hamiltonian_bipartite(cls, hamiltonian, ref_tree) -> StateDiagram:
        """Creates a state diagram equivalent to a given Hamiltonian

        Args:
            hamiltonian (Hamiltonian): Hamiltonian for which the state
                diagram is to be found
            ref_tree (TreeTensorNetwork): Supplies the tree topology which
                is to be incorporated into the state diagram.

        Returns:
            StateDiagram: The final state diagram
        """

        state_diagrams = cls.get_state_diagrams(hamiltonian,ref_tree)
        compound_state_diagram = cls.get_state_diagram_compound(state_diagrams)
        #print(compound_state_diagram)

        #for vert in compound_state_diagram.get_all_vertices():
        #    print(vert.identifier, vert)
        #print("-------------------------------------------------")
        #for hyp in compound_state_diagram.get_all_hyperedges():
        #    print(hyp.hash, hyp)
        
        
        #print("----------------->>>>>>>>>><<<<<<<<<<-----------------")
        
        coeffs_next = [state.coeff for state in state_diagrams]
        queue = deque()

        for child in ref_tree.nodes[ref_tree.root_id].children:
            queue.append((ref_tree.root_id,child))


        while queue:
            
            level_size = len(queue)

            for parent, current_node in queue:
                
                #print("current node" , current_node, " children: ", len(compound_state_diagram.hyperedge_colls[current_node].contained_hyperedges)) 
                
                local_hyperedges = [ ]
                #local_vertices = [ ]
                p_vs = []

                for he in compound_state_diagram.hyperedge_colls[current_node].contained_hyperedges:
                    p_v = he.find_vertex(parent)
                    p_vs.append(p_v)

                    #collected_vertices, collected_edges = cls.traverse_subtree(he,visited=[p_v.identifier])
                    local_hyperedges.append(he)

                    #local_hyperedges.append(collected_edges)
                    #local_vertices.append(collected_vertices)
                
                #print([x.identifier for x in p_vs])
                #print([x.identifier for x in compound_state_diagram.vertex_colls[(parent, current_node)].contained_vertices])
                #print("----- Are they same ??")
                compound_state_diagram.combine_u(local_hyperedges, parent, p_vs)
                                
                #print(len(combined))
                #print(compound_state_diagram)
                #print(combination_info)
                #for hyp in compound_state_diagram.get_all_hyperedges():
                #    print(hyp.hash, hyp)
                        
            for _ in range(level_size):
                parent, current_node = queue.popleft()
                
                #print("current node" , current_node, " parent: ", parent) 
                
                local_vs = copy(compound_state_diagram.hyperedge_colls[parent].contained_hyperedges)
                #print(local_vs)

                compound_state_diagram.combine_v(local_vs, current_node, parent)
                
                ulist = []
                vlist = []
                edges = []
                edge_vertices = compound_state_diagram.get_vertex_coll_two_ids(parent, current_node).contained_vertices

                #print(edge_vertices, "-----")
                
                for vert in edge_vertices:
                    
                    us = vert.get_hyperedges_for_one_node_id(current_node)
                    vs = vert.get_hyperedges_for_one_node_id(parent)
                    
                    for i in range(len(us)):
                        for j in range(len(vs)):
                            edges.append((i + len(ulist),j + len(vlist)))
                    ulist.extend(us)
                    vlist.extend(vs)

                    for u in us:
                        u.vertices.remove(vert)
                    for v in vs:
                        v.vertices.remove(vert)

                
                compound_state_diagram.vertex_colls[(parent,current_node)].contained_vertices = []

                
                #print(ulist) 
                #print(vlist)
                #print(edges)

                bigraph = BipartiteGraph(len(ulist), len(vlist), edges)
                u_cover, v_cover = minimum_vertex_cover(bigraph)
                #print(u_cover, v_cover)



                for i in u_cover:
                    vert = None
                    for j in bigraph.adj_u[i]:
                        #print("connecting:   " , ulist[i], vlist[j])
                        if vert == None:
                            vert = Vertex((parent, current_node), [ulist[i], vlist[j]])
                            compound_state_diagram.vertex_colls[(parent,current_node)].contained_vertices.append(vert)
                            ulist[i].vertices.append(vert)
                            vlist[j].vertices.append(vert)
                        else:
                            vert.hyperedges.append(vlist[j])
                            vlist[j].vertices.append(vert)

                        #print("connected vertices: ---", compound_state_diagram.vertex_colls[(parent,current_node)].contained_vertices)
                        

                        edges.remove((i, j))


                for j in v_cover:
                    vert = None
                    for i in bigraph.adj_v[j]:
                        if (i, j) not in edges:
                            continue

                        #print("V-connecting:   " , ulist[i], vlist[j])
                        if vert == None:
                            vert = Vertex((parent, current_node), [ulist[i], vlist[j]])
                            compound_state_diagram.vertex_colls[(parent,current_node)].contained_vertices.append(vert)
                            ulist[i].vertices.append(vert)
                            vlist[j].vertices.append(vert)
                        else:
                            vert.hyperedges.append(ulist[i])
                            ulist[i].vertices.append(vert)
                        
                    edges.remove((i, j))
                
                for child in ref_tree.nodes[current_node].children:
                    queue.append((current_node,child))

                #print(compound_state_diagram)
                            
            #process_state_diagrams = new_state_diagrams
            #print(len(new_state_diagrams))
            #print(new_state_diagrams)
            #print("----------------------------")       
            

        return compound_state_diagram

    @classmethod
    def sum_states(cls, s1, s2,ref_tree):
        
        state_diag= cls(ref_tree)

        for n1,h1 in s1.hyperedge_colls.items():
            if n1 in s2.hyperedge_colls:
                state_diag.hyperedge_colls[n1] = HyperEdgeColl(n1,h1.contained_hyperedges + s2.hyperedge_colls[n1].contained_hyperedges)
            else:
                state_diag.hyperedge_colls[n1] = h1
        for n2, h2 in s2.hyperedge_colls.items():
            if n2 not in state_diag.hyperedge_colls:
                state_diag.hyperedge_colls[n2] = h2


        for n1,h1 in s1.vertex_colls.items():
            if n1 in s2.vertex_colls:
                state_diag.vertex_colls[n1] = VertexColl(n1,h1.contained_vertices + s2.vertex_colls[n1].contained_vertices)
            else:
                state_diag.vertex_colls[n1] = h1
        for n2, h2 in s2.vertex_colls.items():
            if n2 not in state_diag.vertex_colls:
                state_diag.vertex_colls[n2] = h2

        return state_diag

    def erase_subtree(self, start_edge, erased=None):
        if erased is None:
            # To Do : Apply hash method here
            erased = []  # Tracks visited nodes to avoid cycles
        
        #visited.append(start_edge)
        
        #print(start_edge)
        
        for i in range(len(self.hyperedge_colls[start_edge.corr_node_id].contained_hyperedges)):
            if self.hyperedge_colls[start_edge.corr_node_id].contained_hyperedges[i].identifier == start_edge.identifier:
                self.hyperedge_colls[start_edge.corr_node_id].contained_hyperedges.pop(i)
                break


        #print(start_edge.vertices)
        for vertex in start_edge.vertices:
            # Check if the vertex has been visited to avoid processing the same edge multiple times
            if vertex not in erased:
                
                erased.append(vertex)
                self.vertex_colls[vertex.corr_edge].contained_vertices.remove(vertex)

                for edge in vertex.hyperedges:
                    if edge != start_edge:
                        
                        self.erase_subtree(edge, erased)

    """@classmethod
    def traverse_subtree(cls, start_edge, visited=None, collected_vertices=None, collected_edges=None):
        if visited is None:
            # To Do : Apply hash method here
            visited = []  # Tracks visited nodes to avoid cycles
        if collected_vertices is None:
            collected_vertices = []  # List to collect nodes in the subtree
        if collected_edges is None:
            collected_edges = []  # List to collect hyperedges in the subtree

        #visited.append(start_edge)
        collected_edges.append(start_edge)

        for vertex in start_edge.vertices:
            # Check if the vertex has been visited to avoid processing the same edge multiple times
            if vertex.identifier not in visited:
                collected_vertices.append(vertex)
                visited.append(vertex.identifier)
                for edge in vertex.hyperedges:
                    if edge not in collected_edges:
                        cls.traverse_subtree(edge, visited, collected_vertices, collected_edges)

        return collected_vertices, collected_edges"""

    def combine_v(self, local_vs, current_node, parent):
        combined = set()

        for i, element1 in enumerate(local_vs):
            if i in combined:
                continue
            for j in range(i+1,len(local_vs)) :
                element2 = local_vs[j]
                if j in combined:
                    continue

                if element1.label == element2.label:
                    same = True
                    for v in element1.vertices:
                        if not(v.corr_edge == (current_node,parent) or v.corr_edge == (parent,current_node)):
                            if not v in element2.vertices:
                                same = False
                                break
                    
                    
                    if same and len(element1.vertices) == len(element2.vertices) :


                        #print(i,j)
                        #print(element1.label,element2.label)

                        

                        d1 = len(element1.find_vertex(current_node).get_hyperedges_for_one_node_id(parent)) > 1
                        d2 = len(element2.find_vertex(current_node).get_hyperedges_for_one_node_id(parent)) > 1

                        if d1 and d2:
                            continue

                        combined.add(j)

                        son = None
                        son2 = None

                        del_vertex = element2.find_vertex(current_node)
                        for h in del_vertex.hyperedges:
                            if h.corr_node_id == current_node:
                                son = h

                        other_vertex = element1.find_vertex(current_node)
                        for h in other_vertex.hyperedges:
                            if h.corr_node_id == current_node:
                                son2 = h
                        
                                

                        if not (d1 or d2):
                            for vert in element2.vertices:
                                if vert.corr_edge == (current_node,parent) or vert.corr_edge == (parent,current_node):
                                    # Del_vertex from the vertex collection
                                    self.vertex_colls[(parent,current_node)].contained_vertices.remove(vert)

                                    # Add son to element 1 vertex collection
                                    element1.find_vertex(current_node).add_hyperedge(son)

                                    # Remove del_vertex from the son hyperedge
                                    son.vertices.remove(vert)
                                else:
                                    # Just delete hyperedge from other vertices
                                    #vert.hyperedges.remove(element2)

                                    for i in range(len(vert.hyperedges)):
                                        if vert.hyperedges[i].identifier == element2.identifier:
                                            vert.hyperedges.pop(i)
                                            break

                            
                            # Remove Hyperedge from state diagram        
                            #self.hyperedge_colls[element2.corr_node_id].contained_hyperedges.remove(element2)
                            self._remove_hyperedge(element2)
                            

                        elif d1:
                            for vert in element2.vertices:
                                if vert.corr_edge == (current_node,parent) or vert.corr_edge == (parent,current_node):
                                    # Del_vertex from the vertex collection
                                    self.vertex_colls[(parent,current_node)].contained_vertices.remove(vert)

                                    # Remove del_vertex from the son hyperedge
                                    son.vertices.remove(vert)

                                    # Create new hyperedge
                                    new_h = HyperEdge(current_node, son2.label, [])
                                    new_h.set_hash(son2.hash)

                                    # Add vertices to new hyperedge unrelated to current node-parent
                                    for v in son2.vertices:
                                        if not(v.corr_edge == (current_node,parent) or v.corr_edge == (parent,current_node)):
                                            new_h.add_vertex(v)
                                    
                                    # Create a new vertex
                                    new_v = Vertex((parent, current_node), [new_h])


                                    deleted_hyperedges = []
                                    # Add new hyperedges to the vertex
                                    for h in other_vertex.hyperedges:
                                        if h.identifier != son2.identifier and h.identifier != element1.identifier :
                                            #print("visiting other vertice hyperedges: ", h.identifier, h.vertices)
                                            
                                            try:
                                                h.vertices.remove(other_vertex)
                                            except:
                                                print("Hamiltonian: ", self.hamiltonian)
                                                
                                                raise ValueError("Error")
                                            deleted_hyperedges.append(h)
                                            new_v.add_hyperedge(h)
                                            #print("vertices after adding: ", h.identifier, h.vertices)

                                    while len(deleted_hyperedges) > 0:
                                        for i in range(len(other_vertex.hyperedges)):
                                            if other_vertex.hyperedges[i].identifier == deleted_hyperedges[0].identifier:
                                                other_vertex.hyperedges.pop(i)
                                                deleted_hyperedges.pop(0)
                                                break
                                    


                                    self.vertex_colls[(parent,current_node)].contained_vertices.append(new_v)
                                    new_h.vertices.append(new_v)

                                    self.add_hyperedge(new_h)

                                    # Add son to element 1 vertex collection
                                    other_vertex.add_hyperedge(son)

                                    
                                else:
                                    # Just delete hyperedge from other vertices
                                    for i in range(len(vert.hyperedges)):
                                        if vert.hyperedges[i].identifier == element2.identifier:
                                            vert.hyperedges.pop(i)
                                            break

                            self._remove_hyperedge(element2)
                            

                        elif d2:
                            for vert in element2.vertices:
                                if vert.corr_edge == (current_node,parent) or vert.corr_edge == (parent,current_node):
                                    # Del_vertex from the vertex collection
                                    self.vertex_colls[(parent,current_node)].contained_vertices.remove(vert)

                                    # Remove del_vertex from the son hyperedge
                                    son.vertices.remove(vert)

                                    # Create new hyperedge
                                    new_h = HyperEdge(current_node, son.label, [])
                                    new_h.set_hash(son.hash)

                                    # Add vertices to new hyperedge unrelated to current node-parent
                                    for v in son.vertices:
                                        if not(v.corr_edge == (current_node,parent) or v.corr_edge == (parent,current_node)):
                                            new_h.add_vertex(v)
                                    
                                    # Create new vertex
                                    new_v = Vertex((parent, current_node), [new_h])


                                    # Add new hyperedges to the vertex
                                    for h in del_vertex.hyperedges:
                                        if h.identifier != son.identifier and h.identifier != element2.identifier :
                                            h.vertices.remove(del_vertex)
                                            new_v.add_hyperedge(h)

                                    self.vertex_colls[(parent,current_node)].contained_vertices.append(new_v)
                                    new_h.vertices.append(new_v)

                                    self.add_hyperedge(new_h)
                                    

                                    # Add son to element 1 vertex collection
                                    other_vertex.add_hyperedge(son)

                                    
                                else:
                                    # Just delete hyperedge from other vertices
                                    for i in range(len(vert.hyperedges)):
                                        if vert.hyperedges[i].identifier == element2.identifier:
                                            vert.hyperedges.pop(i)
                                            break
                                    
                            self._remove_hyperedge(element2)


                        #print(self)

    def combine_u(self, local_hyperedges, parent, p_vs):
        combined = set()

        for i, element1 in enumerate(local_hyperedges):
            if i in combined:
                continue
            for j in range(i+1,len(local_hyperedges)) :
                element2 = local_hyperedges[j]
                if j in combined:
                    continue

                if element1.hash == element2.hash:

                    #print(i,j)
                    combined.add(j)
                    
                    #print("element to delete: " , element2)
                    self.erase_subtree(element2, erased=[p_vs[j]])

                    #print(self)

                    fathers = []
                    for h in p_vs[j].hyperedges:
                        if h.corr_node_id == parent:
                            fathers.append(h)

                    #print("before remove: ", p_vs[j].identifier,[x.identifier for x in self.vertex_colls[p_vs[j].corr_edge].contained_vertices]) 
                    self.vertex_colls[p_vs[j].corr_edge].contained_vertices.remove(p_vs[j])
                
                    for father in fathers:
                        father.vertices.remove(p_vs[j])

                   
                    #print(p_vs[i],p_vs[i].identifier)
                    #print([x.identifier for x in self.vertex_colls[p_vs[j].corr_edge].contained_vertices])

                    for father in fathers:
                        p_vs[i].add_hyperedge(father)

                    #for vert in self.vertex_colls[p_vs[j].corr_edge].contained_vertices:
                        #print(vert, vert.identifier)
                    #    if vert.hyperedges ==  p_vs[i].hyperedges:
                    #        vert.add_hyperedge(father)

    def calculate_hashes(self, node, ref_tree):
        # Base case: if the node is None, just return
        if node is None:
            return ""
        # Process all children first
        children_hash = ""
        for child_id in node.children:
            children_hash += self.calculate_hashes(ref_tree.nodes[child_id], ref_tree)
        # Process the current node (parent node is processed after its children)
        return self.hyperedge_colls[node.identifier].contained_hyperedges[0].calculate_hash(children_hash) 

    @classmethod
    def get_state_diagram_compound(cls, state_diagrams):
        
        state_diagram = None

        for term in state_diagrams:
            if state_diagram != None:
                state_diagram = cls.sum_states(state_diagram, term, term.reference_tree)
            else:
                state_diagram = term
            
            
        return state_diagram

    @classmethod
    def get_state_diagrams(cls, hamiltonian, ref_tree):
        
        state_diagrams = []

        for term in hamiltonian.terms:
            state_diagram = cls.from_single_term(term, ref_tree)
            state_diagrams.append(state_diagram)

            state_diagram.calculate_hashes(ref_tree.nodes[ref_tree.root_id],ref_tree)
            
        return state_diagrams

    def _remove_hyperedge(self, element):
        for i in range(len(self.hyperedge_colls[element.corr_node_id].contained_hyperedges)):
            if self.hyperedge_colls[element.corr_node_id].contained_hyperedges[i].identifier == element.identifier:
                self.hyperedge_colls[element.corr_node_id].contained_hyperedges.pop(i)
                break