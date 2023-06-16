from __future__ import annotations

import numpy as np

from .tree_structure import TreeStructure, Node
from .canonical_form import canonical_form
from .tree_contraction import (completely_contract_tree,
                               contract_two_ttn,
                               single_site_operator_expectation_value,
                               operator_expectation_value,
                               scalar_product
                               )

class TreeTensorNetwork(TreeStructure):
    """
    A tree tensor network (TTN) a tree, where each node contains a tensor,
    that is part of the network. Here a tree tensor network is a dictionary
    _nodes of tensor nodes with their identifiers as keys.

    General structure and parts of the codes are from treelib.tree

    Attributes
    -------
    _nodes: dict[str, TensorNode] mapping node ids (str) to TensorNode objects
    _root_id: str identifier for root node of TTN
    """

    def __init__(self):
        """
        Initiates a new TreeTensorNetwork or a deep or shallow copy of a
        different one.
        """
        super().__init__()
        self._tensors = {}

    @property
    def tensors(self):
        """
        A dict[str, ndarray] mapping the tensor tree node identifiers to
        the corresponding tensor data.

        Since during addition of nodes the tensors are not actually transposed,
        this has to be done here. This way whenever tensors are accessed, their
        leg ordering is
            (parent_leg, children_legs, open_legs)
        """
        new_tensors = {}
        for node_id, tensor in self._tensors.items():
            node = self.nodes[node_id]
            transposed_tensor = np.transpose(tensor, node.leg_permutation)
            new_tensors[node_id] = transposed_tensor
            node.reset_permutation()

        self._tensors = new_tensors
        return self._tensors

    def add_root(self, node, tensor):
        """
        Adds a root tensor node to the TreeTensorNetwork
        """
        super().add_root(node)
        self.tensors[node.identifier] = tensor

    def add_child_to_parent(
            self, child: Node, tensor: ndarray, child_leg: int, parent_id: str, parent_leg: int):
        """
        Adds a Node to the TreeTensorNetwork which is the child of the Node
        with identifier `parent_id`. The two tensors are contracted along one
        leg. The child via child_leg and the parent via parent_leg
        """
        self._tensors[child.identifier] = tensor
        super().add_child_to_parent(child, child_leg, parent_id, parent_leg)

    def add_parent_to_root(self, root_leg: int, parent: Node, tensor: ndarray, parent_leg: int):
        """
        Adds the node parent as parent to the TreeTensorNetwork's root node. The two
        are contracted. The root via root_leg and the parent via parent_leg.
        The root is updated to be the parent.
        """
        super().add_parent_to_root(root_leg, parent, parent_leg)
        self.tensors[parent.identifier] = tensor

    def conjugate(self):
        """
        Returns a new TTN that is a conjugated version of the current TTN

        Returns
        -------
        ttn_conj:
            A conjugated copy of the current TTN.

        """

        ttn_conj = TreeTensorNetwork(original_tree=self, deep=True)

        for node_id in ttn_conj.nodes:

            node = ttn_conj.nodes[node_id]
            node.tensor = np.conj(node.tensor)

        return ttn_conj

    # Functions below this are just wrappers of external functions that are
    # linked tightly to the TTN and its structure. This allows these functions
    # to be overwritten for subclasses of the TTN with more known structure.
    # The additional sturcture allows for more efficent algorithms than the
    # general case.

    def canonical_form(self, orthogonality_center_id):
        """
        Brings the tree_tensor_network in canonical form with

        Parameters
        ----------
        orthogonality_center_id : str
            The id of the tensor node, which sould be the orthogonality center for
            the canonical form

        Returns
        -------
        None.
        """
        canonical_form(self, orthogonality_center_id)

    def completely_contract_tree(self, to_copy: bool = False):
        """
        Completely contracts the given tree_tensor_network by combining all
        nodes.
        (WARNING: Can get very costly very fast. Only use for debugging.)

        Parameters
        ----------
        to_copy: bool
            Wether or not the contraction should be perfomed on a deep copy.
            Default is False.

        Returns
        -------
        In case copy is True a deep copy of the completely contracted TTN is
        returned.

        """
        return completely_contract_tree(self, to_copy=to_copy)

    def contract_two_ttn(self, other):
        """
        Contracts two TTN with the same structure. Assumes both TTN use the same
        identifiers for the nodes.

        Parameters
        ----------
        other : TreeTensorNetwork

        Returns
        -------
        result_tensor: ndarray
            The contraction result.

        """
        return contract_two_ttn(self, other)

    def single_site_operator_expectation_value(self, node_id, operator):
        """
        Assuming ttn represents a quantum state, this function evaluates the
        expectation value of the operator applied to the node with identifier
        node_id.

        Parameters
        ----------
        node_id : string
            Identifier of a node in ttn.
            Currently assumes the node has a single open leg..
        operator : ndarray
            A matrix representing the operator to be evaluated.

        Returns
        -------
        exp_value: complex
            The resulting expectation value.

        """
        return single_site_operator_expectation_value(self, node_id,
                                                      operator)

    def operator_expectation_value(self, operator_dict):
        """
        Assuming ttn represents a quantum state, this function evaluates the
        expectation value of the operator.

        Parameters
        ----------
        operator_dict : dict
            A dictionary representing an operator applied to a quantum state.
            The keys are node identifiers to which the value, a matrix, is applied.

        Returns
        -------
        exp_value: complex
            The resulting expectation value.

        """
        return operator_expectation_value(self, operator_dict)

    def scalar_product(self):
        """
        Computes the scalar product for a state_like TTN, i.e. one where the open
        legs represent a quantum state.

        Parameters
        ----------
        None

        Returns
        -------
        sc_prod: complex
            The resulting scalar product.

        """
        return scalar_product(self)

    def absorb_tensor(self, node_id: str, absorbed_tensor: ndarray, absorbed_tensors_leg_indices: tuple[int],
                      this_tensors_leg_indices: tuple[int]):
        """
        Absorbs `absorbed_tensor` into this instance's tensor by contracting
        the absorbed_tensors_legs of the absorbed_tensor and the legs
        this_tensors_legs of this instance's tensor'

        Parameters
        ----------
        absorbed_tensor: ndarray
            Tensor to be absorbed.
        absorbed_tensors_leg_indices: int or tuple of int
            Legs that are to be contracted with this instance's tensor.
        this_tensors_leg_indices:
            The legs of this instance's tensor that are to be contracted with
            the absorbed tensor.
        """
        if type(absorbed_tensors_leg_indices) == int:
            absorbed_tensors_leg_indices = (absorbed_tensors_leg_indices, )

        if type(this_tensors_leg_indices) == int:
            this_tensors_leg_indices = (this_tensors_leg_indices, )

        assert len(absorbed_tensors_leg_indices) == len(this_tensors_leg_indices)

        if len(absorbed_tensors_leg_indices) == 1:
            this_tensors_leg_index = this_tensors_leg_indices[0]
            new_tensor = np.tensordot(self.tensors[node_id], absorbed_tensor,
                                      axes=(this_tensors_leg_indices, absorbed_tensors_leg_indices))

            this_tensors_indices = tuple(range(new_tensor.ndim))
            transpose_perm = (this_tensors_indices[0:this_tensors_leg_index]
                              + (this_tensors_indices[-1], )
                              + this_tensors_indices[this_tensors_leg_index:-1])
            self.tensors[node_id] = new_tensor.transpose(transpose_perm)
        else:
            raise NotImplementedError

    def apply_hamiltonian(self, hamiltonian: Hamiltonian, conversion_dict: dict[str, ndarray], skipped_vertices=None):
        """
        Applies a Hamiltonian term by term locally to a TTN. Assumes that the input TTN represents a statevector
        such that each TensorNode has the following memory layout: [parent, child_1, child_2, ..., child_n, output].
        """

        def allocate_output_tensor(node_id, state_diagram, conversion_dict):
            """
            Allocates output tensor for each node
            """
            he = state_diagram.hyperedge_colls[node_id].contained_hyperedges[0]
            operator_label = he.label
            operator = conversion_dict[operator_label]
            # Should be square operators
            phys_dim = operator.shape[0]

            total_tensor_shape = [0] * len(he.vertices)
            slice_tensor_shape = [0] * len(he.vertices)
            total_tensor_shape.extend([phys_dim])
            slice_tensor_shape.extend([phys_dim])
            node = self.nodes[node_id]
            neighbours = node.neighbouring_nodes()
            for leg_index, neighbour_id in enumerate(neighbours.keys()):
                vertex_coll = state_diagram.get_vertex_coll_two_ids(
                    node_id, neighbour_id)
                for index_value, vertex in enumerate(vertex_coll.contained_vertices):
                    vertex.index = (leg_index, index_value)
                # The number of vertices is equal to the number of bond-dimensions required.
                total_tensor_shape[leg_index] = len(
                    vertex_coll.contained_vertices) * self.tensors[node_id].shape[neighbours[neighbour_id]]
                slice_tensor_shape[leg_index] = self.tensors[node_id].shape[neighbours[neighbour_id]]
            output_tensor = np.zeros(total_tensor_shape, dtype=np.cdouble)
            return output_tensor, slice_tensor_shape

        from .ttno.state_diagram import StateDiagram
        state_diagram = StateDiagram.from_hamiltonian(hamiltonian, self)

        # Adding the operator corresponding to each hyperedge to the tensor
        for node_id, hyperedge_coll in state_diagram.hyperedge_colls.items():
            output_tensor, output_slice_shape = allocate_output_tensor(
                node_id, state_diagram, hamiltonian.conversion_dictionary)
            local_tensor = self._nodes[node_id]

            for he in hyperedge_coll.contained_hyperedges:
                index_value = [0] * len(he.vertices)

                for vertex in he.vertices:
                    index_value[vertex.index[0]] = vertex.index[1]

                slice_indexing = [slice(index * size, (index+1) * size, 1)
                                  for (index, size) in zip(index_value, output_slice_shape)]
                slice_indexing.extend([slice(None)])
                slice_indexing = tuple(slice_indexing)
                operator_label = he.label
                operator = hamiltonian.conversion_dictionary[operator_label]
                output_slice = np.tensordot(
                    self.tensors[local_tensor.identifier], operator, axes=([local_tensor.nlegs()-1], [1]))
                output_tensor[slice_indexing] += output_slice

            self.tensors[node_id] = output_tensor
