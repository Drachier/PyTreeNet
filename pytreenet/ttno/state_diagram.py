from __future__ import annotations
from typing import Dict, Tuple, List
from copy import copy
from collections import deque
from enum import Enum

import numpy as np

from ..core.tree_structure import TreeStructure
from ..operators.hamiltonian import Hamiltonian
from ..util.ttn_exceptions import NoConnectionException, NotCompatibleException
from .vertex import Vertex
from .hyperedge import HyperEdge
from .collections import VertexColl, HyperEdgeColl
from .single_term_diagram import SingleTermDiagram


class TTNOFinder(Enum):
    """
    An Enum to switch between different construction modes of a state diagram.
    """
    TREE = "Tree"
    CM = "Combine and Match"


class StateDiagram():
    """ 
    A state diagram represents a Hamiltonian (or other operator)

    In principle it can represent any tree tensor network. It contains vertices
    and hyperedges, that can be sorted into an underlying tree structure.

    Attributes:
        vertex_colls (VertexColl): A collection of all vertices in the state
            diagram. They are grouped by the edge that corresponds to them.
        hyperedge_colls (HyperEdgeColl): A collection of all hyperedges in the
            state diagram. They are grouped by the node that corresponds to
            them.
        reference_tree (TreeStructure): Provides the underlying tree structure
            with connectivity and identifiers.
    """

    def __init__(self, reference_tree: TreeStructure):
        """
        Initialises a state diagram.

        Args:
            reference_tree (TreeStructure): Provides the underlying tree
                structure with connectivity and identifiers.
        """
        self.vertex_colls: Dict[Tuple[str, str], VertexColl] = {}
        self.hyperedge_colls: Dict[str, HyperEdgeColl] = {}
        self.reference_tree = reference_tree

    def __repr__(self) -> str:
        """
        A string representation of a state diagram.
        """
        all_he = self.get_all_hyperedges()
        all_vert = self.get_all_vertices()
        string = "hyperedges:\n"
        for hyperedge in all_he:
            string += str(hyperedge) + "\n"
        string += "\n vertices:\n"
        for vertex in all_vert:
            string += str(vertex) + "\n"

        return string

    def get_all_vertices(self) -> List[Vertex]:
        """
        Returns all vertices from all collections in a list.
        """
        all_vert = []
        for vertex_coll in self.vertex_colls.values():
            all_vert.extend(vertex_coll.contained_vertices)
        return all_vert

    def get_all_hyperedges(self) -> List[HyperEdge]:
        """
        Returns all hyperedges from all collections in a list
        """
        all_he = []
        for hyperedge_coll in self.hyperedge_colls.values():
            all_he.extend(hyperedge_coll.contained_hyperedges)
        return all_he

    def get_vertex_coll_two_ids(self,
                                id1: str,
                                id2: str) -> VertexColl:
        """
        Obtain the vertex collection corresponding to a specified edge.

        The edge is specified by the two node identifiers given and is the edge
        connecting the two. Since the order of the identifiers in the edge is
        irrelevant, the same is true for the supplied identifiers.

        Args:
            id1 (str): One identifier corresponding a node connected by the
                edge.
            id2 (str): One identifier corresponding a node connected by the
                edge.

        Returns:
            VertexColl: The vertex collection corresponding to the specified
                edge.
        """
        key1 = (id1, id2)
        key2 = (id2, id1)
        if key1 in self.vertex_colls:
            return self.vertex_colls[key1]
        if key2 in self.vertex_colls:
            return self.vertex_colls[key2]
        errstr = f"There is no vertex collection corresponding to and edge between {id1} and {id2}!"
        raise NoConnectionException(errstr)

    def add_hyperedge(self, hyperedge: HyperEdge):
        """
        Adds a hyperedge to the correct collection and to the state diagram.

        Args:
            hyperedge (HyperEdge): The hyperedge to be added.
        """
        node_id = hyperedge.corr_node_id

        if node_id in self.reference_tree.nodes:
            self.hyperedge_colls[node_id].contained_hyperedges.append(
                hyperedge)
        else:
            errstr = f"No node with identifier {node_id} in reference tree!"
            raise NotCompatibleException(errstr)

    @classmethod
    def from_hamiltonian(cls,
                         hamiltonian: Hamiltonian,
                         ref_tree: TreeStructure,
                         method: TTNOFinder = TTNOFinder.TREE) -> StateDiagram:
        """
        Creates a state diagram equivalent to a given Hamiltonian.

        Args:
            hamiltonian (Hamiltonian): Hamiltonian for which the state
                diagram is to be found
            ref_tree (TreeTensorNetwork): Supplies the tree topology which
                is to be incorporated into the state diagram.
            method (TTNOFinder): The construction method to be used.

        Returns:
            StateDiagram: The final state diagram

        """
        state_diagram = None
        if method == method.CM:
            state_diagram = cls.from_hamiltonian_combine_match(hamiltonian,
                                                               ref_tree)
        elif method == method.TREE:
            state_diagram = cls.from_hamiltonian_tree_comparison(hamiltonian,
                                                                 ref_tree)
        else:
            errstr = "Invalid construction method!"
            raise ValueError(errstr)
        return state_diagram

    @classmethod
    def from_hamiltonian_tree_comparison(cls,
                                         hamiltonian: Hamiltonian,
                                         ref_tree: TreeStructure):
        """
        Constructs a Hamiltonian using the leaf to root comparison method.

        Args:
            hamiltonian (Hamiltonian): Hamiltonian for which the state
                diagram is to be found
            ref_tree (TreeTensorNetwork): Supplies the tree topology which
                is to be incorporated into the state diagram.

        Returns:
            StateDiagram: The final state diagram.

        """
        state_diagram = None
        for term in hamiltonian.terms:
            if state_diagram is None:
                state_diagram = cls.from_single_term(term, ref_tree)
            else:
                state_diagram.add_single_term(term)
        return state_diagram

    @classmethod
    def from_single_term(cls,
                         term: Dict[str, str],
                         reference_tree: TreeStructure):
        """
        Basically a wrap of ``SingleTermDiagram.from_single_term``.
        """
        single_term_diagram = SingleTermDiagram.from_single_term(term,
                                                                 reference_tree)
        return cls.from_single_state_diagram(single_term_diagram)

    @classmethod
    def from_single_state_diagram(cls,
                                  single_term_diag: SingleTermDiagram) -> StateDiagram:
        """
        Transforms a single state diagram to a general one.

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

    def add_single_term(self, term: Dict[str, str]):
        """
        Modifies the state diagram to add a term.

        Adds a term to the state diagram. This means the diagram is modified in
        a way such that it represents Hamiltonian + term instead of only
        the Hamiltonian.

        Args:
            term (Dict[str,str]): A dictionary containing the node_ids as keys
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
                self._find_new_he(next_vertex, new_node_id,
                                  single_term_diagram)

    def _find_and_mark_new_vertex(self, current_hyperedge):
        current_node_id = current_hyperedge.corr_node_id
        uncontained_vertex = current_hyperedge.get_single_uncontained_vertex()
        next_node_id = uncontained_vertex.get_second_node_id(current_node_id)
        if not current_hyperedge.vertex_single_he(next_node_id):
            # In this case we would add multiple paths
            return None
        vertex_coll = self.get_vertex_coll_two_ids(
            current_node_id, next_node_id)
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
        Find the shape of the tensor corresponding to a node in the equivalent
        TTNO.

        Args:
            node_id (str): The identifier of a node.
            conversion_dict (Dict[str, np.ndarray]): A dictionary to convert
                the labels into arrays, to determine the required physical
                dimension.

        Returns:
            Tuple[int, ...]: The shape of the tensor in the equivalent TTNO in
                the format 
                ``(parent_shape, children_shape, phys_dim, phys_dim)``.
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
        Indexes all vertices contained in this state diagram.

        This index is the index value to which this vertex corresponds in the
        bond dimension.
        """
        for vertex_coll in self.vertex_colls.values():
            vertex_coll.index_vertices()

    def reset_markers(self):
        """
        Resets the contained and new markers of every vertex in the diagram.
        """
        for vertex_col in self.vertex_colls.values():
            for vertex in vertex_col.contained_vertices:
                vertex.contained = False
                vertex.new = False

    @classmethod
    def from_hamiltonian_combine_match(cls,
                                       hamiltonian: Hamiltonian,
                                       ref_tree: TreeStructure) -> StateDiagram:
        """
        Creates optimal state diagram equivalent to a given Hamiltonian.

        Args:
            hamiltonian (Hamiltonian): Hamiltonian for which the state
                diagram is to be found
            ref_tree (TreeTensorNetwork): Supplies the tree topology which
                is to be incorporated into the state diagram.

        Returns:
            StateDiagram: The final state diagram
        """

        # Get individual state diagrams and combine them into a compound state diagram
        state_diagrams = cls.get_state_diagrams(hamiltonian, ref_tree)
        compound_state_diagram = cls.get_state_diagram_compound(state_diagrams)

        # For the future implementations:
        # coeffs_next = [state.coeff for state in state_diagrams]

        # Queue for tree traversal. We traverse the tree in a BFS manner
        queue = deque()

        # Add all children of the root node to the queue
        for child in ref_tree.nodes[ref_tree.root_id].children:
            queue.append((ref_tree.root_id, child))

        while queue:

            # For each level of the tree, we need to combine the hyperedges on both current and parent level.
            # This requires a two-step process. First, we combine the hyperedges on the current level. ( U nodes )
            # Then, we combine the hyperedges on the parent level. ( V nodes )

            # Level size is the number of nodes on the current level. This allows us to pass twice.
            level_size = len(queue)

            # Combining hyperedges on the current level
            for parent, current_node in queue:

                # Combining u nodes
                local_hyperedges = copy(
                    compound_state_diagram.hyperedge_colls[current_node].contained_hyperedges)
                compound_state_diagram.combine_u(local_hyperedges, parent)

            # Combining hyperedges on the parent level
            for _ in range(level_size):

                # After second pass, we pop the element from the queue
                parent, current_node = queue.popleft()

                # Combining v nodes
                local_vs = copy(
                    compound_state_diagram.hyperedge_colls[parent].contained_hyperedges)
                compound_state_diagram.combine_v(
                    local_vs, current_node, parent)

                # Add all children of the current node to the queue (BFS traversal)
                for child in ref_tree.nodes[current_node].children:
                    queue.append((current_node, child))

        return compound_state_diagram

    # TODO: Refactor to be shorter.
    def combine_v(self, local_vs, current_node, parent):
        """
        Checks if the hyperedges in the local_vs list can be combined as v nodes 
        and when it is possible, combines them. Combining v nodes also means combining
        vertices in the cut site.

        There are 4 cases to consider when combining v nodes with respect to 
        the their vertices in the cut site:

        1. Both vertices have more than one hyperedge to the parent node. (d1 and d2)
        2. Both vertices have only one hyperedge to the parent node. (not d1 and not d2)
        3. One vertex has more than one hyperedge to the parent node, the other has only one. (not d1 and d2) 
        4. One vertex has only one hyperedge to the parent node, the other has more than one. (d1 and not d2) 

        Case 1:
        In this case, we only check if there are a fully matching set of hyperedges between the vertices.
        If we can form a fully connected vertex, we combine the vertices and remove the hyperedges and call the function again.
        If not, keep both hyperedges as they are.

        Case 2:
        In this case, we basically remove the second hyperedge and its connections and
        combine the vertices (as connecting second one to the first one).

        Case 3:
        In this case, we handle the vertex with more than one hyperedge to the parent node specially.
        We create a new vertex and duplicate hyperedges and seperate connections to the new vertex.
        Then, we combine the vertices.

        Case 4:
        Same as case 3, so we just switch element1 and element2.

        After iterating each pair of hyperedges, we check if there are any combined hyperedges.
        If there are, we call the function recursively to check the hyperedges again.
        """
        combined = set()
        generated = False

        for i, element1 in enumerate(local_vs):
            if i in combined:
                continue

            for j in range(i+1, len(local_vs)):

                if j in combined:
                    continue

                element2 = local_vs[j]

                # Checking if two hyperedges are suitable for combining as v nodes.
                # The check conditions are stated in the _is_same_v function.
                if StateDiagram._is_same_v(element1, element2, current_node, parent):

                    # vertices of the elements in the cut site
                    keep_vertex = element1.find_vertex(current_node)
                    del_vertex = element2.find_vertex(current_node)

                    # d1 and d2 boolean values are used to check if the vertices of the elements
                    # have more than one hyperedge to the parent node in the cut site.
                    # They are used to determine the case of the combination.
                    d1 = keep_vertex.num_hyperedges_to_node(parent) > 1
                    d2 = del_vertex.num_hyperedges_to_node(parent) > 1

                    # del_sons and keep_sons are the hyperedges of the vertices in the cut site.
                    del_sons = del_vertex.get_hyperedges_for_one_node_id(
                        current_node)
                    keep_sons = keep_vertex.get_hyperedges_for_one_node_id(
                        current_node)

                    # Case 1
                    if d1 and d2:
                        # Check if it is possible to create a fully connected vertex. If not, simply continue.
                        if not StateDiagram._is_fully_connected(del_vertex, keep_vertex, current_node, parent):
                            continue
                        else:

                            # Create a fully connected vertex and delete all duplicate hyperedges.
                            del_hyperedges_parent = del_vertex.get_hyperedges_for_one_node_id(
                                parent)
                            for element in del_hyperedges_parent:
                                self._connect_two_vertices(
                                    element, current_node, parent, del_sons, keep_vertex)

                            # Call the function recursively to check the hyperedges again.
                            # As we removed a lot of hyperedges at once, we need to check the hyperedges again from beginning.
                            local_vs = copy(
                                self.hyperedge_colls[parent].contained_hyperedges)
                            self.combine_v(local_vs, current_node, parent)
                            return

                    # Keep track of combined hyperedges
                    combined.add(j)

                    # Case 2
                    # Combine two vertices and remove the second hyperedge.
                    if not (d1 or d2):
                        self._connect_two_vertices(
                            element2, current_node, parent, del_sons, keep_vertex)

                    else:
                        # Case 4
                        # Switch element1 and element2
                        generated = True
                        if d1:

                            element1, element2 = element2, element1
                            del_sons, keep_sons = keep_sons, del_sons
                            del_vertex, keep_vertex = keep_vertex, del_vertex

                        # Case 3
                        for vert in element2.vertices:

                            # Handle vertex at the cut site.
                            if vert.corr_edge == (current_node, parent) or vert.corr_edge == (parent, current_node):

                                # Delete vert from the vertex collection
                                self.vertex_colls[(
                                    parent, current_node)].contained_vertices.remove(vert)

                                # Create new hyperedges as duplicating del_sons
                                new_hs = []
                                for son in del_sons:
                                    new_h = HyperEdge(
                                        current_node, son.label, [])
                                    new_h.set_hash(son.hash)
                                    new_hs.append(new_h)

                                    # Add vertices to new hyperedge unrelated to current node-parent
                                    for v in son.vertices:
                                        if not (v.corr_edge == (current_node, parent) or v.corr_edge == (parent, current_node)):
                                            new_h.add_vertex(v)

                                # Create a new vertex
                                new_v = Vertex(
                                    (parent, current_node), new_hs.copy())

                                # This new vertex is connected to every hyperedge of the del_vertex
                                # except the del_sons and element2 ( which means hyperedges in the parent site except element2).
                                identifiers = [
                                    x.identifier for x in del_sons] + [element2.identifier]
                                for h in del_vertex.hyperedges:
                                    if h.identifier not in identifiers:
                                        h.vertices.remove(del_vertex)
                                        new_v.add_hyperedge(h)

                                # Add new vertex to the state diagram
                                self.vertex_colls[(parent, current_node)].contained_vertices.append(
                                    new_v)

                                # Add new hyperedges to the state diagram and connect with new vertex
                                for new_h in new_hs:
                                    new_h.vertices.append(new_v)
                                    self.add_hyperedge(new_h)

                                # Connect del_sons to keep_vertex
                                for son in del_sons:
                                    son.vertices.remove(del_vertex)
                                    keep_vertex.add_hyperedge(son)

                            else:
                                # Just remove hyperedge from other vertices
                                self._remove_hyperedge(vert, element2)

                        self._remove_hyperedge_from_diagram(element2)

        if generated:
            local_vs = copy(self.hyperedge_colls[parent].contained_hyperedges)
            self.combine_v(local_vs, current_node, parent)
            return

        return

    def combine_u(self, local_hyperedges, parent):
        """
        Checks if the hyperedges in the local_hyperedges list can be combined 
        and when it is possible, combines them.

        Combining two hyperedge is basically removing one of the subtree of the hyperedge
        and connecting deleted subtree's vertex to the remaining hyperedge.
        """
        combined = set()

        for i, element1 in enumerate(local_hyperedges):
            if i in combined:
                continue

            for j in range(i+1, len(local_hyperedges)):
                if j in combined:
                    continue

                element2 = local_hyperedges[j]

                # Checking if two hyperedges are suitable for combining.
                # It is enough to check hashes as hashes are containing unique information
                # of the hyperedge and all of the children hyperedges till leaves, aka subtree.
                if element1.hash == element2.hash:

                    # Keep track of combined hyperedges
                    combined.add(j)

                    # Find the vertex to be deleted and the vertex to be kept.
                    del_vertex = element2.find_vertex(parent)
                    keep_vertex = element1.find_vertex(parent)

                    # Erase the subtree of the element2 hyperedge from the state diagram
                    self.erase_subtree(element2, erased=[del_vertex])

                    # Find the hyperedges that the del_vertex is connected to in the parent site.
                    fathers = del_vertex.get_hyperedges_for_one_node_id(parent)

                    # Remove the del_vertex from the vertex collection
                    self.vertex_colls[del_vertex.corr_edge].contained_vertices.remove(
                        del_vertex)

                    # Reconnect hyperedges of the del_vertex on the parent site to the keep_vertex.
                    for father in fathers:
                        father.vertices.remove(del_vertex)
                        keep_vertex.add_hyperedge(father)

    def erase_subtree(self, start_edge, erased=None):
        """
        Erases the subtree of a hyperedge from the state diagram recursively.
        """

        if erased is None:
            erased = []  # Tracks visited nodes to avoid cycles

        # Remove the hyperedge from the hyperedge collection
        for i in range(len(self.hyperedge_colls[start_edge.corr_node_id].contained_hyperedges)):
            if self.hyperedge_colls[start_edge.corr_node_id].contained_hyperedges[i].identifier == start_edge.identifier:
                self.hyperedge_colls[start_edge.corr_node_id].contained_hyperedges.pop(
                    i)
                break

        for vertex in start_edge.vertices:
            # Check if the vertex has been visited to avoid processing the same edge multiple times
            if vertex not in erased:

                erased.append(vertex)
                self.vertex_colls[vertex.corr_edge].contained_vertices.remove(
                    vertex)

                for edge in vertex.hyperedges:
                    if edge != start_edge:
                        self.erase_subtree(edge, erased)

    @classmethod
    def get_state_diagram_compound(cls, state_diagrams):
        """
        Forms a compound state diagram from a list of state diagrams.
        """

        state_diagram = None

        for term in state_diagrams:
            if state_diagram != None:
                state_diagram = cls.sum_states(
                    state_diagram, term, term.reference_tree)
            else:
                state_diagram = term

        return state_diagram

    @classmethod
    def sum_states(cls, s1, s2, ref_tree):
        """
        Combines two state diagrams into a single one and returns it. 
        """

        state_diag = cls(ref_tree)

        for n1, h1 in s1.hyperedge_colls.items():
            if n1 in s2.hyperedge_colls:
                state_diag.hyperedge_colls[n1] = HyperEdgeColl(
                    n1, h1.contained_hyperedges + s2.hyperedge_colls[n1].contained_hyperedges)
            else:
                state_diag.hyperedge_colls[n1] = h1
        for n2, h2 in s2.hyperedge_colls.items():
            if n2 not in state_diag.hyperedge_colls:
                state_diag.hyperedge_colls[n2] = h2

        for n1, h1 in s1.vertex_colls.items():
            if n1 in s2.vertex_colls:
                state_diag.vertex_colls[n1] = VertexColl(
                    n1, h1.contained_vertices + s2.vertex_colls[n1].contained_vertices)
            else:
                state_diag.vertex_colls[n1] = h1
        for n2, h2 in s2.vertex_colls.items():
            if n2 not in state_diag.vertex_colls:
                state_diag.vertex_colls[n2] = h2

        return state_diag

    @classmethod
    def get_state_diagrams(cls, hamiltonian, ref_tree):
        """
        Creates single term diagrams for each term in the Hamiltonian.
        Calculates hash values for each state diagram.
        """

        state_diagrams = []

        for term in hamiltonian.terms:
            state_diagram = cls.from_single_term(term, ref_tree)
            state_diagrams.append(state_diagram)

            state_diagram.calculate_hashes(
                ref_tree.nodes[ref_tree.root_id], ref_tree)

        return state_diagrams

    def calculate_hashes(self, node, ref_tree):
        """
        Calculates and returns the hash of all of the hyperedges in the state diagram recursively. 
        Hash is formed by the label of the hyperedge and concatenation of the hashes of its children.
        """
        # Base case: if the node is None, just return
        if node is None:
            return ""
        # Process all children first
        children_hash = ""
        for child_id in node.children:
            children_hash += self.calculate_hashes(
                ref_tree.nodes[child_id], ref_tree)
        # Process the current node (parent node is processed after its children)
        return self.hyperedge_colls[node.identifier].contained_hyperedges[0].calculate_hash(children_hash)

    def _remove_hyperedge_from_diagram(self, element):
        """
        Removes a hyperedge from the state diagram checking its identifier.
        """
        for i in range(len(self.hyperedge_colls[element.corr_node_id].contained_hyperedges)):
            if self.hyperedge_colls[element.corr_node_id].contained_hyperedges[i].identifier == element.identifier:
                self.hyperedge_colls[element.corr_node_id].contained_hyperedges.pop(
                    i)
                break

    @classmethod
    def _remove_hyperedge(cls, vert, element):
        for i in range(len(vert.hyperedges)):
            if vert.hyperedges[i].identifier == element.identifier:
                vert.hyperedges.pop(i)
                break

    @classmethod
    def _is_same_v(cls, element1, element2, current_node, parent):
        """ 
        Checks two hyperedges for suitability for combining as v nodes.
        Checks their labels and all vertices except cut vertex (current_node,parent) for equality.
        Returns true if they are suitable for combining, false otherwise.
        """

        if element1.label == element2.label:
            same = True
            for v in element1.vertices:
                if not (v.corr_edge == (current_node, parent) or v.corr_edge == (parent, current_node)):
                    if not v in element2.vertices:
                        same = False
                        break

            return same and len(element1.vertices) == len(element2.vertices)
        return False

    @classmethod
    def _is_fully_connected(cls, del_vertex, keep_vertex, current_node, parent):
        """
        Checks del_vertex and keep_vertex for forming a fully connected vertex in the cut site (current_node-parent).
        Returns true if they can form a fully connected vertex, false otherwise.
        """
        if del_vertex.num_hyperedges_to_node(parent) == keep_vertex.num_hyperedges_to_node(parent):
            first_set = del_vertex.get_hyperedges_for_one_node_id(parent)
            second_set = keep_vertex.get_hyperedges_for_one_node_id(parent)
            for h1 in first_set:
                h_match = False
                for h2 in second_set:
                    same = False
                    if h1.label == h2.label:
                        same = True
                        for v in h1.vertices:
                            if not (v.corr_edge == (current_node, parent) or v.corr_edge == (parent, current_node)):
                                if not v in h2.vertices:
                                    same = False
                                    break

                        if same and len(h1.vertices) == len(h2.vertices):
                            h_match = True
                            break
                if not h_match:
                    return False
            return True
        else:
            return False

    def _connect_two_vertices(self, element, current_node, parent, del_sons, keep_vertex):
        """
        Connects del_sons to the keep_vertex and removes the del_vertex from the state diagram in the cut site.
        Removes element from each of its vertices and removes element from the state diagram.
        """
        for vert in element.vertices:
            if vert in self.vertex_colls[(parent, current_node)].contained_vertices:
                # Remove vert from the compound state diagram
                self.vertex_colls[(parent, current_node)
                                  ].contained_vertices.remove(vert)

                # Connect del_sons to keep_vertex
                for son in del_sons:
                    # Remove vert from the son hyperedge
                    son.vertices.remove(vert)
                    keep_vertex.add_hyperedge(son)
            else:

                # Just delete hyperedge from other vertices

                self._remove_hyperedge(vert, element)

        # Remove Hyperedge from the state diagram
        self._remove_hyperedge_from_diagram(element)
