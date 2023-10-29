from __future__ import annotations
from typing import Dict, Tuple
from copy import copy

from .vertex import Vertex
from .hyperedge import HyperEdge
from .collections import VertexColl, HyperEdgeColl
from .single_term_diagram import SingleTermDiagram


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
    def from_hamiltonian(cls, hamiltonian, ref_tree):
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
        neighbours = self.reference_tree[node_id].neighbouring_nodes()
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
        for vertex_coll in self.vertex_colls:
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
