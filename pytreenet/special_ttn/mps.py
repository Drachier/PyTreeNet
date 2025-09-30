"""
Provides the matrix product topology TTN.

A matrix product topology is just a chain of nodes. Nodes can be added
automatically on either side. This is the most commonly used topology with MPS
and MPO. These are also supplied combining the matrix product topology with the
properties of a TTNS or TTNO.
"""
from __future__ import annotations
from typing import List, Union, Any, Callable
from copy import deepcopy

import numpy as np

from ..core.ttn import TreeTensorNetwork
from ..ttns import TreeTensorNetworkState
from ..core.node import Node
from ..ttno.ttno_class import TTNO
from ..util.ttn_exceptions import non_negativity_check
from ..operators.common_operators import ket_i
from .util import check_product_state_parameters

class MatrixProductTree(TreeTensorNetwork):
    """
    A tree tensor network in the form of a chain.

    In principle every node can have an arbitrary number of legs. Important
    special cases are the MPS and MPO. Mostly used for testing.

    Attributes:
        left_nodes (Dict[str, Node]): A dictionary of nodes that are attached
            to the leftmost node. The keys are the identifiers of the nodes.
        right_nodes (Dict[str, Node]): A dictionary of nodes that are attached
            to the rightmost node. The keys are the identifiers of the nodes.
    """

    def __init__(self):
        """
        Intialises a constant matrix product tree/chain.

        Args:
            left_nodes (List[Node], optional): A list of nodes that are
                attached to the left of the root node. The order in the
                list is the same order as in the chain. Defaults to [].
            right_nodes (List[Node], optional): A list of nodes that are
                attached to the right of the root node. The order in the
                list is the same order as in the chain. Defaults to [].
        """
        super().__init__()
        self.left_nodes: List[Node] = []
        self.right_nodes: List[Node] = []

    @classmethod
    def from_tensor_list(cls, tensor_list: List[np.ndarray],
                         node_prefix: str | list[str] = "site",
                         root_site: int = 0) -> Any:
        """
        Generates a MatrixProductTree from a list of tensors.
        
        The nodes in the MPT will be considered as from left to right, in the
        same way as they are in the list.

        Args:
            tensor_list (List[np.ndarray]): A list with site tensors. Their
                legs should be of the form
                    `[left_leg,right_leg,open_legs]`
            node_prefix (str | list[str], optional): Either a prefix that
                should be part of the node identifiers before the site index,
                or a list of identifiers to be used for the nodes. If a list
                is provided, its length must be the same as the number of
                tensors. Defaults to "site".
            root_site (int, optional): Which tensor should be associated to
                the root node. Defaults to 0.

        Returns:
            MatrixProductTree: A matrix product tree representing an MP
                structure A_1 * A_2  ... A_N, where the A are the tensors in
                the provided list.
        """
        if isinstance(node_prefix, list):
            if len(node_prefix) != len(tensor_list):
                errstr = "If a list of node identifiers is provided, its "
                errstr += "length must be the same as the number of tensors!"
                errstr += f" Got {len(node_prefix)} and {len(tensor_list)}."
                raise ValueError(errstr)
            def id_func(i: int) -> str:
                return node_prefix[i]
        else:
            def id_func(i: int) -> str:
                return node_prefix + str(i)
        non_negativity_check(root_site, "root site")
        if root_site >= len(tensor_list):
            errstr = "Root site must be an available site!"
            raise ValueError(errstr)
        mpt = cls()
        mpt.add_root(Node(identifier=id_func(root_site)),
                     tensor_list[root_site])
        if root_site==0:
            return cls.from_tensor_list_leftmost_node_is_root(tensor_list,
                                                              id_func)
        left_tensors = reversed(tensor_list[0:root_site])
        for i, tensor in enumerate(left_tensors):
            site = root_site - 1 - i
            mpt.attach_node_left_end(Node(identifier=id_func(site)),
                                     tensor,
                                     final=site==0)
        right_tensors = tensor_list[root_site+1:]
        for i, tensor in enumerate(right_tensors):
            site = root_site + 1 + i
            mpt.attach_node_right_end(Node(identifier=id_func(site)),
                                      tensor)
        return mpt

    @classmethod
    def from_tensor_list_leftmost_node_is_root(cls,
                                               tensor_list: List[np.ndarray],
                                               node_id_map: Callable) -> Any:
        """
        Generates a MatrixProductTree from a list of tensors, where the
        leftmost tensor, i.e. index 0, corresponds to the root node.
        
        The nodes in the MPT will be considered as from left to right, in the
        same way as they are in the list.

        Args:
            tensor_list (List[np.ndarray]): A list with site tensors. Their
                legs should be of the form
                    `[left_leg,right_leg,open_legs]`
            node_id_map (Callable): A function that takes an integer index
                and returns a string identifier for the node at that index.

        Returns:
            MatrixProductTree: A matrix product tree representing an MP
                structure A_1 * A_2  ... A_N, where the A are the tensors in
                the provided list.
        """
        mpt = cls()
        mpt.add_root(Node(identifier=node_id_map(0)),
                     tensor_list[0])
        if len(tensor_list)>1:
            node1 = Node(identifier=node_id_map(1))
            mpt.add_child_to_parent(node1,
                                    tensor_list[1],0,node_id_map(0),
                                    0)
            mpt.right_nodes.append(node1)
        if len(tensor_list)>2:
            remaining_tensors = tensor_list[2:]
            for i, tensor in enumerate(remaining_tensors):
                site = i + 2
                mpt.attach_node_right_end(Node(identifier=node_id_map(site)),
                                            tensor)
        return mpt

    def attach_node_right_end(self, node: Node, tensor: np.ndarray):
        """
        Attaches a node as a child to the rightmost node.

        Args:
            node (Node): The node to be added.
            tensor (np.ndarray): The tensor to be associated with the node.
             Legs should be of the form
              `[parent_leg, other_virtual_leg, open_legs]`
        """
        if len(self.right_nodes) == 0:
            parent_node = self.root[0]
            parent_leg = 1
        else:
            parent_node = self.right_nodes[-1]
            parent_leg = 1
        self.add_child_to_parent(node, tensor, 0,
                                 parent_node.identifier, parent_leg)
        self.right_nodes.append(node)

    def attach_node_left_end(self, node: Node, tensor: np.ndarray,
                             final: bool = False):
        """
        Attaches a node as a child to the leftmost node.

        Args:
            node (Node): The node to be added.
            tensor (np.ndarray): The tensor to be associated with the node.
                Legs should be of the form
                    `[other_virtual_leg, parent_leg, open_legs]`
        """
        if len(self.left_nodes) == 0:
            parent_node = self.root[0]
            parent_leg = len(parent_node.children)
        else:
            parent_node = self.left_nodes[0]
            parent_leg = 1
        self.add_child_to_parent(node, tensor, int(not final),
                                 parent_node.identifier, parent_leg)
        self.left_nodes.insert(0, node)

class MatrixProductState(MatrixProductTree, TreeTensorNetworkState):
    """
    A state with the matrix product topology.

    A matrix product state (MPS) is one of the most common forms of tensor
    networks. It is a chain of tensors, each with one physical leg.
    """

    @classmethod
    def constant_product_state(cls, state_value: int,
                               dimension: int,
                               num_sites: int,
                               node_prefix: str = "site",
                               root_site: int = 0,
                               bond_dimensions: Union[List[int],None] = None) -> Any:
        """
        Generates an MPS that corresponds to a product state with the same
        value at every site.

        Args:
            state_value (int): The state's value at every site.
            dimension (int): The local dimension of the MPS.
            num_sites (int): The number of sites in the MPS.
            node_prefix (str, optional): A prefix that should be part of the
                node identifiers before the site index. Defaults to "site".
            root_site (int, optional): Which tensor should be associated to
                the root node. Defaults to 0.
            bond_dimensions (Union[List[int],None]): Give custom bond
                dimensions. The zeroth entry will be the dimension between
                nodes 0 and 1 and so forth. Defaults to None, in which case
                the bond dimensions are all one.

        Returns:
            MatrixProductState: MPS representing a product state of the form
                |psi> = |value> otimes |value> otimes ... |value>
        """
        check_product_state_parameters(state_value,dimension)
        non_negativity_check(num_sites, "number of sites")
        single_site_tensor = ket_i(state_value, dimension)
        single_site_tensor = np.reshape(single_site_tensor, (1,1,dimension))
        tensor_list = [deepcopy(single_site_tensor[0,:,:])]
        tensor_list.extend([deepcopy(single_site_tensor) for _ in range(num_sites-2)])
        tensor_list.append(deepcopy(single_site_tensor[0,:,:]))
        if bond_dimensions is not None:
            if len(bond_dimensions) != num_sites-1:
                errstr = "There must be as many bond dimensions as bonds!"
                raise ValueError(errstr)
            first_tensor = np.pad(deepcopy(tensor_list[0]),
                                  [(0,bond_dimensions[0]-1),(0,0)])
            new_tensors = [first_tensor]
            padded_tensors = [np.pad(deepcopy(tensor),
                                     [(0,bond_dimensions[i]-1),
                                      (0,bond_dimensions[i+1]-1),(0,0)])
                              for i, tensor in enumerate(tensor_list[1:-1])]
            new_tensors.extend(padded_tensors)
            final_tensor = np.pad(deepcopy(tensor_list[-1]),
                                  [(0,bond_dimensions[-1]-1),(0,0)])
            new_tensors.append(final_tensor)
            tensor_list = new_tensors
        return cls.from_tensor_list(tensor_list,
                                    node_prefix=node_prefix,
                                    root_site=root_site)

    def __str__(self) -> str:
        """
        Returns a string representation of the MPS.
        
        The leftmost node is at the top and the rightmost node at the bottom.
        If the MPS is empty, returns "Empty MPS".
        """
        string = ""
        for node in self.left_nodes:
            string += f"{node.identifier} : {node.shape}\n"
        if self.root_id is not None:
            string += f"{self.root_id} : {self.root[1].shape}\n"
        else:
            return "Empty MPS"
        for node in self.right_nodes:
            string += f"{node.identifier} : {node.shape}\n"
        return string

class MatrixProductOperator(MatrixProductTree, TTNO):
    pass
