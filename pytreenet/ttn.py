from __future__ import annotations
from typing import Tuple, Dict, List, Callable
from copy import copy, deepcopy
from collections import UserDict

import numpy as np

from .tree_structure import TreeStructure
from .node import Node
from .tensor_util import (tensor_qr_decomposition,
                          contr_truncated_svd_splitting)
from .leg_specification import LegSpecification
from .canonical_form import canonical_form
from .tree_contraction import (completely_contract_tree,
                               contract_two_ttn)


class TensorDict(UserDict):
    def __init__(self, nodes, inpt=None):
        if inpt is None:
            inpt = {}
        super().__init__(inpt)
        self.nodes = nodes

    def __getitem__(self, node_id: str):
        """
        Since during addition of nodes the tensors are not actually transposed,
        this has to be done when accesing them. 
        This way whenever a tensor is accessed, its leg ordering is
            (parent_leg, children_legs, open_legs)
        """
        permutation = self.nodes[node_id].leg_permutation
        tensor = super().__getitem__(node_id)
        transposed_tensor = np.transpose(tensor, permutation)
        self.nodes[node_id]._reset_permutation()
        super().__setitem__(node_id, transposed_tensor)
        return transposed_tensor


class TreeTensorNetwork(TreeStructure):
    """
    A tree tensor network (TTN) a tree, where each node contains a tensor,
    that is part of the network. Here a tree tensor network is a dictionary
    _nodes of tensor nodes with their identifiers as keys.

    General structure and parts of the codes are from treelib.tree

    Attributes
    -------
    _nodes: dict[str, Node] mapping node ids (str) to Node objects
    _tensors: dict[str, ndarray] mapping node ids (str) to numpy ndarray objects
    _root_id: str identifier for root node of TTN
    """

    def __init__(self):
        """
        Initiates a new TreeTensorNetwork or a deep or shallow copy of a
        different one.
        """
        super().__init__()
        self._tensors = TensorDict(self._nodes)

    @property
    def tensors(self):
        """
        A dict[str, np.ndarray] mapping the tensor tree node identifiers to
        the corresponding tensor data.

        Since during addition of nodes the tensors are not actually transposed,
        this has to be done here. This way whenever tensors are accessed, their
        leg ordering is
            (parent_leg, children_legs, open_legs)
        """

        return self._tensors

    def _transpose_tensor(self, node_id: str):
        """
        Since during addition of nodes the tensors are not actually transposed,
        this has to be done when accesing them. 
        This way whenever a tensor is accessed, its leg ordering is
            (parent_leg, children_legs, open_legs)
        """
        node = self.nodes[node_id]
        tensor = self._tensors[node_id]
        transposed_tensor = np.transpose(tensor, node.leg_permutation)
        self._tensors[node_id] = transposed_tensor
        node.reset_permutation()

    def __getitem__(self, key: str) -> Tuple[Node, np.ndarray]:
        node = super().__getitem__(key)
        tensor = self._tensors[key]
        return (node, tensor)

    def add_root(self, node: Node, tensor: np.ndarray):
        """
        Adds a root tensor node to the TreeTensorNetwork
        """
        node.link_tensor(tensor)
        super().add_root(node)

        self.tensors[node.identifier] = tensor

    def add_child_to_parent(self, child: Node, tensor: np.ndarray,
                            child_leg: int, parent_id: str, parent_leg: int):
        """
        Adds a Node to the TreeTensorNetwork which is the child of the Node
        with identifier `parent_id`. The two tensors are contracted along one
        leg; the child via child_leg and the parent via parent_leg
        """
        self.ensure_existence(parent_id)
        child.link_tensor(tensor)
        self._add_node(child)
        child.open_leg_to_parent(parent_id, child_leg)

        child_id = child.identifier
        parent_node = self._nodes[parent_id]
        parent_node.open_leg_to_child(child_id, parent_leg)

        self.tensors[child_id] = tensor

    def add_parent_to_root(self, root_leg: int, parent: Node, tensor: np.ndarray,
                           parent_leg: int):
        """
        Adds the Node `parent` as parent to the TreeTensorNetwork's root node. The two
        nodes are connected: the root via root_leg and the parent via parent_leg.
        The root is updated to be the parent.
        """
        self._add_node(parent)
        parent.open_leg_to_child(self.root_id, parent_leg)
        new_root_id = parent.identifier
        former_root_node = self.nodes[self.root_id]
        former_root_node.open_leg_to_parent(new_root_id, root_leg)
        self._root_id = new_root_id
        self.tensors[new_root_id] = tensor

    def conjugate(self):
        """
        Returns a new TTN that is a conjugated version of the current TTN

        Returns
        -------
        ttn_conj:
            A conjugated copy of the current TTN.

        """
        ttn_conj = deepcopy(self)
        for node_id, tensor in ttn_conj.tensors.items():
            ttn_conj.tensors[node_id] = tensor.conj()
        return ttn_conj

    def absorb_tensor(self, node_id: str, absorbed_tensor: np.ndarray,
                      absorbed_tensors_leg_index: int,
                      this_tensors_leg_index: int):
        """
        Absorbs `absorbed_tensor` into this instance's tensor by contracting
        the absorbed_tensors_leg of the absorbed_tensor and the leg
        this_tensors_leg of this instance's tensor'

        Parameters
        ----------
        absorbed_tensor: np.ndarray
            Tensor to be absorbed.
        absorbed_tensors_leg_index: int
            Leg that is to be contracted with this instance's tensor.
        this_tensors_leg_index:
            The leg of this instance's tensor that is to be contracted with
            the absorbed tensor.
        """
        node_tensor = self.tensors[node_id]
        new_tensor = np.tensordot(node_tensor, absorbed_tensor,
                                  axes=(this_tensors_leg_index, absorbed_tensors_leg_index))

        this_tensors_indices = tuple(range(new_tensor.ndim))
        transpose_perm = (this_tensors_indices[0:this_tensors_leg_index]
                          + (this_tensors_indices[-1], )
                          + this_tensors_indices[this_tensors_leg_index:-1])
        self.tensors[node_id] = new_tensor.transpose(transpose_perm)

    def absorb_tensor_into_neighbour_leg(self, node_id: str, neighbour_id: str,
                                         tensor: np.ndarray, tensor_leg: int):
        """
        Absorb a tensor into a node, by contracting one of the tensor's legs with one of the
        neighbour_legs of the node.

        Args:
            node_id (str): The identifier of the node into which the tensor is absorbed
            neighbour_id (str): The identifier of the neighbour to which the leg points, which
                                 is to be contracted with the tensor
            tensor (np.ndarray): The tensor to be contracted
            tensor_leg (int): The leg of the external tensor which is to be contracted
        """
        assert tensor.ndim == 2
        node = self.nodes[node_id]
        neighbour_leg = node.get_neighbour_leg(neighbour_id)
        self.absorb_tensor(node_id, tensor, tensor_leg, neighbour_leg)

    def absorb_into_open_legs(self, node_id: str, tensor: np.ndarray):
        """
        Absorb a tensor into the open legs of the tensor of a node.
        This tensor will be absorbed into all open legs and it is assumed, the
         leg order of the tensor to be absorbed is the same as the order of
         the open legs of the node.
        The tensor to be absorbed has to have twice as many open legs as the node tensor.
         The first half of the legs is contracted with the tensor node's open legs and
         the second half become the new open legs of the tensor node.

        Args:
            node_id (str): The identifier of the node which is to be contracted with the tensor
            tensor (np.ndarray): The tensor to be contracted.
        """
        node, node_tensor = self[node_id]
        assert tensor.ndim == 2 * node.nopen_legs()

        tensor_legs = list(range(node.nopen_legs()))
        new_tensor = np.tensordot(node_tensor, tensor, axes=(node.open_legs, tensor_legs))
        # The leg ordering was not changed here
        self.tensors[node_id] = new_tensor

    def contract_nodes(self, node_id1: str, node_id2: str, new_identifier: str = ""):
        """
        Contracts two node and inserts a new node with the contracted tensor
        into the ttn.
        Note that one of the nodes will be the parent of the other.
        The resulting leg order is the following:
            `(parent_parent_leg, node1_children_legs, node2_children_legs,
            node1_open_legs, node2_open_legs)`
        The resulting node will have the identifier `parent_id + "contr" + child_id`.

        Deletes the original nodes and tensors from the TTN.

        Args:
            node_id1 (str): Identifier of first tensor
            node_id2 (str): Identifier of second tensor
            new_identifier (str): A potential new identifier. Otherwise defaults to
                `parent_id + 'contr' + child_id`
        """
        parent_id, child_id = self.determine_parentage(node_id1, node_id2)
        if new_identifier == "":
            new_identifier = parent_id + "contr" + child_id

        child_node = self.nodes[child_id]
        parent_node = self.nodes[parent_id]

        # Contracting tensors
        parent_tensor = self.tensors[parent_id]
        child_tensor = self.tensors[child_id]
        print(parent_tensor.shape, child_tensor.shape)
        print(parent_node.children, child_node.children)
        new_tensor = np.tensordot(parent_tensor, child_tensor,
                                  axes=(parent_node.get_child_leg(child_id), 0))

        # remove old tensors
        self.tensors.pop(node_id1)
        self.tensors.pop(node_id2)

        # add new tensor
        self.tensors[new_identifier] = new_tensor
        new_node = Node(tensor=new_tensor, identifier=new_identifier)

        # Actual tensor leg now have the form
        # (parent_of_parent, remaining_children_of_parent, open_of_parent,
        # children_of_child, open_of_child)
        if not parent_node.is_root():
            new_node.open_leg_to_parent(parent_node.parent, 0)
        parent_children = copy(parent_node.children)
        parent_children.remove(child_id)
        parent_child_dict = {identifier: leg_value + parent_node.nparents()
                             for leg_value, identifier in enumerate(parent_children)}
        child_children_dict = {identifier: leg_value + parent_node.nlegs() - 1
                               for leg_value, identifier in enumerate(child_node.children)}
        print(parent_child_dict)
        print(child_children_dict)
        if parent_id == node_id1:
            parent_child_dict.update(child_children_dict)
            new_node.open_legs_to_children(parent_child_dict)
            # new_node.open_legs_to_children(child_children_dict)
        else:
            child_children_dict.update(parent_child_dict)
            new_node.open_legs_to_children(child_children_dict)
            # new_node.open_legs_to_children(parent_child_dict)
        if node_id1 != parent_id:
            new_nvirt = new_node.nvirt_legs()
            range_parent = range(new_nvirt, new_nvirt + parent_node.nopen_legs())
            range_child = range(new_nvirt + parent_node.nopen_legs(), new_node.nlegs())
            new_node.exchange_open_leg_ranges(range_parent, range_child)

        # Change connectivity
        self.replace_node_in_neighbours(new_identifier, parent_id)
        self.replace_node_in_neighbours(new_identifier, child_id)
        self._nodes[new_identifier] = new_node
        print(new_node.shape, new_node.children, new_node.leg_permutation, new_tensor.shape)

    def legs_before_combination(self, node1_id: str, node2_id: str) -> Tuple[LegSpecification, LegSpecification]:
        """
        When combining two nodes, the information about their legs is lost.
         However, sometimes one wants to split the two nodes again, as they were
         before. This function provides the required leg specification for the
         splitting.

        Args:
            node1_id (str): _description_
            node2_id (str): _description_

        Returns:
            Tuple[LegSpecification, LegSpecification]: _description_
        """

        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        tot_nvirt_legs = node1.nvirt_legs() + node2.nvirt_legs() - 2
        tot_nlegs = node1.nlegs() + node2.nlegs() - 2
        open_legs1 = list(range(tot_nvirt_legs, tot_nvirt_legs + node1.nopen_legs()))
        open_legs2 = list(range(tot_nvirt_legs + node1.nopen_legs(), tot_nlegs))
        spec1 = LegSpecification(parent_leg=None,
                                 child_legs=copy(node1.children),
                                 open_legs=open_legs1,
                                 node=None)
        spec2 = LegSpecification(parent_leg=None,
                                 child_legs=copy(node2.children),
                                 open_legs=open_legs2,
                                 node=None)
        temp = [(spec1, node1), (spec2, node2)]
        if node2.is_parent_of(node1_id):
            temp.reverse()
        temp[0][0].parent_leg = temp[0][1].parent
        temp[0][0].child_legs.remove(temp[1][1].identifier)

    def _split_nodes(self, node_id: str, out_legs: Dict[str, List], in_legs: Dict[str, List],
                     splitting_function: Callable, out_identifier: str = "", in_identifier: str= "",
                     **kwargs):
        """
        Splits an node into two nodes using a specified function

        Args:
            node_id (str): The identifier of the node to be split.
            out_legs (Dict[str, List]): The legs associated to the output of the matricised
                node tensor. (The Q legs for QR and U legs for SVD)
            in_legs (Dict[str, List]): The legs associated to the input of the matricised
                node tensor: (The R legs for QR and the SVh legs for SVD)
            splitting_function (Callable): The function to be used for the splitting
            out_identifier (str, optional): An identifier for the tensor with the output
                legs. Defaults to "".
            in_identifier (str, optional): An identifier for the tensor with the input
                legs. Defaults to "".
            **kwargs: Are passed to the splitting function.
        """
        node, tensor = self[node_id]
        if isinstance(out_legs, dict):
            out_legs = LegSpecification.from_dict(out_legs, node)
        elif out_legs.node is None:
            out_legs.node = node
        if isinstance(in_legs, dict):
            in_legs = LegSpecification.from_dict(in_legs, node)
        elif in_legs.node is None:
            in_legs.node = node

        # Find new identifiers
        if out_identifier == "":
            out_identifier = "out_of_" + node_id
        if in_identifier == "":
            in_identifier = "in_of_" + node_id

        # Getting the numerical value of the legs
        out_legs_int = out_legs.find_leg_values()
        in_legs_int = in_legs.find_leg_values()
        out_tensor, in_tensor = splitting_function(tensor, out_legs_int, in_legs_int, **kwargs)
        self._tensors[out_identifier] = out_tensor
        self._tensors[in_identifier] = in_tensor

        # New Nodes
        out_node = Node(tensor=out_tensor, identifier=out_identifier)
        in_node = Node(tensor=in_tensor, identifier=in_identifier)
        self._nodes[out_identifier] = out_node
        self._nodes[in_identifier] = in_node

        # Currently the tensors out and in have the leg ordering
        # (new_leg(for in), parent_leg, virtual_leg, open_legs, new_leg(for out))
        if in_legs.parent_leg is not None:
            in_node.open_leg_to_parent(in_legs.parent_leg, 1)
            in_children = {child_id: leg_value + 2
                           for leg_value, child_id in enumerate(in_legs.child_legs)}
            in_children[out_identifier] = 1
            in_node.open_legs_to_children(in_children)
            out_node.open_leg_to_parent(in_identifier, out_node.nlegs() - 1)
            out_children = {child_id: leg_value + 1
                            for leg_value, child_id in enumerate(out_legs.child_legs)}
            out_node.open_legs_to_children(out_children)
        else:
            # For the in_tensor all legs are in the correct position
            in_node.open_leg_to_parent(out_identifier, 0)
            in_children = {child_id: leg_value + 1
                           for leg_value, child_id in enumerate(in_legs.child_legs)}
            in_node.open_legs_to_children(in_children)
            if node.is_root():
                self._root_id = out_identifier
            else:
                out_node.open_leg_to_parent(out_legs.parent_leg, 0)
            out_children = {child_id: leg_value + out_node.nparents()
                            for leg_value, child_id in enumerate(out_legs.child_legs)}
            out_children[in_identifier] = out_node.nlegs() - 1
            out_node.open_legs_to_children(out_children)

        self.replace_node_in_some_neighbours(out_identifier, node_id,
                                              out_legs.find_all_neighbour_ids())
        self.replace_node_in_some_neighbours(in_identifier, node_id,
                                              in_legs.find_all_neighbour_ids())
        self._nodes.pop(node_id)

    def split_node_qr(self, node_id: str, q_legs: Dict[str, List], r_legs: Dict[str, List],
                      q_identifier: str = "", r_identifier: str = ""):
        """
        Splits a node into two nodes via QR-decomposition.

        The legs should be given as a dictionary with the keys
        "parent_leg", "child_legs" and "open_legs".
        If there is no parent, it should be denoted by None.

        Args:
            node_id (str): The node to be split
            q_legs (Dict[str, List]): The legs which should be part of the Q-tensor
            r_legs (Dict[str, List]): The legs which should be part of the R-tensor
            q_identifier (str, optional): An identifier for the Q-tensor. Defaults to "".
            r_identifier (str, optional): An identifier for the R-tensor. Defaults to "".
        """
        self._split_nodes(node_id, q_legs, r_legs, tensor_qr_decomposition,
                          out_identifier=q_identifier, in_identifier=r_identifier)

    def split_node_svd(self, node_id: str, u_legs: Dict, v_legs: Dict,
                       u_identifier: str = "", v_identifier: str = "",
                       **truncation_param):
        """
        Splits a node in two using singular value decomposition. In the process the tensors
         are truncated as specified by truncation parameters. The singular values
         are absorbed into the v_legs.

        Args:
            node_id (str): Identifier of the nodes to be split
            u_legs (Dict): The legs which should be part of the U tensor
            v_legs (Dict): The legs which should be part of the V tensor
            u_identifier (str): An identifier for the U-tensor. Defaults to "".
            v_identifier (str): An identifier for the V-tensor. Defaults to "".
        """
        self._split_nodes(node_id, u_legs, v_legs, contr_truncated_svd_splitting,
                          out_identifier=u_identifier, in_identifier=v_identifier,
                          **truncation_param)

    # Functions below this are just wrappers of external functions that are
    # linked tightly to the TTN and its structure. This allows these functions
    # to be overwritten for subclasses of the TTN with more known structure.
    # The additional structure allows for more efficent algorithms than the
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
        result_tensor: np.ndarray
            The contraction result.

        """
        return contract_two_ttn(self, other)

    def apply_hamiltonian(
            self, hamiltonian: Hamiltonian, conversion_dict: dict[str, np.ndarray],
            skipped_vertices=None):
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
