# TODO: Implement if the root is site 0

from __future__ import annotations
from typing import List
from copy import deepcopy

import numpy as np

from ..ttn import TreeTensorNetwork
from ..ttns import TreeTensorNetworkState
from ..node import Node
from ..ttno.ttno import TTNO

class MatrixProductTree(TreeTensorNetwork):
    """
    A tree tensor network in the form of a chain.
     In principle every node can have an arbitrary number
     of legs. Important special cases are the MPS and MPO.

     Mostly used for testing.
    """

    def __init__(self):
        """Intialises a constant matrix product tree/chain

        Args:
            node_prefix (str, optional): Nodes will have an identifier of the
             form `"node_prefix "+ number`. Defaults to `"site"`.
            first_site (int, optional): The number that should be associated to
             the leftmost site. Defaults to `0`.
        """
        super().__init__()

        self.left_nodes = {}
        self.right_nodes = {}

    @classmethod
    def from_tensor_list(cls, tensor_list: List[np.ndarray],
                         node_prefix: str = "site",
                         root_site: int = 0) -> MatrixProductTree:
        """
        Generates a MatrixProductTree from a list of tensors. The nodes in the
         MPT will be considered as from left to right, in the same way as they
         are in the list.

        Args:
            tensor_list (List[np.ndarray]): A list with site tensors. Their
             legs should be of the form
              `[left_leg,right_leg,open_legs]`
            node_prefix (str, optional): A prefix that should be part of the
             node identifiers before the site index. Defaults to "site".
            root_site (int, optional): Which tensor should be associated to
             the root node. Defaults to 0.

        Returns:
            MatrixProductTree: A matrix product tree representing an MP
             structure A_1 * A_2  ... A_N, where the A are the tensors in the
             provided list.
        """
        mpt = cls()
        mpt.add_root(Node(identifier=node_prefix+str(root_site)),
                     tensor_list[root_site])

        left_tensors = reversed(tensor_list[0:root_site])
        for i, tensor in enumerate(left_tensors):
            site = root_site - 1 - i
            mpt.attach_node_left_end(Node(identifier=node_prefix+str(site)),
                                     tensor,
                                     final=site==0)
        right_tensors = tensor_list[root_site+1:]
        for i, tensor in enumerate(right_tensors):
            site = root_site + 1 + i
            mpt.attach_node_right_end(Node(identifier=node_prefix+str(site)),
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
        if not self.right_nodes:
            parent_node = self.root[0]
            parent_leg = 1
        else:
            parent_node = list(self.right_nodes.values())[-1]
            parent_leg = 1
        self.add_child_to_parent(node, tensor, 0,
                                 parent_node.identifier, parent_leg)
        self.right_nodes[node.identifier] = node

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
        if not self.left_nodes:
            parent_node = self.root[0]
            parent_leg = len(parent_node.children)
        else:
            parent_node = list(self.left_nodes.values())[-1]
            parent_leg = 1
        self.add_child_to_parent(node, tensor, not final,
                                 parent_node.identifier, parent_leg)
        self.left_nodes[node.identifier] = node

class MatrixProductState(MatrixProductTree, TreeTensorNetworkState):

    @classmethod
    def constant_product_state(cls, state_value: int,
                               dimension: int,
                               num_sites: int,
                               node_prefix: str = "site",
                               root_site: int = 0) -> MatrixProductState:
        """
        Generates an MPS that corresponds to a product state with the same value
            at every site.

        Args:
            state_value (int): The state's value at every site.
            dimension (int): The local dimension of the MPS.
            num_sites (int): The number of sites in the MPS.
            node_prefix (str, optional): A prefix that should be part of the
             node identifiers before the site index. Defaults to "site".
            root_site (int, optional): Which tensor should be associated to
             the root node. Defaults to 0.

        Raises:
            ValueError: If state_value is negative or dimension non-positive.
                Also raised if state value is larger than dimension.

        Returns:
            MatrixProductState: MPS representing a product state of the form
                |psi> = |value> otimes |value> otimes ... |value>
        """
        if dimension < 1:
            errstr = f"Dimension of a state must be positive not {dimension}!"
            raise ValueError(errstr)
        if state_value >= dimension:
            errstr = "State value cannot be larger than the state's dimension!"
            raise ValueError(errstr)
        elif state_value < 0:
            errstr = f"State value must be non-negative not {state_value}!"
            raise ValueError(errstr)
        if num_sites < 1:
            errstr = f"Number of sites must be positive not {num_sites}!"
            raise ValueError(errstr)
        single_site_tensor = np.zeros(dimension)
        single_site_tensor[state_value] = 1
        single_site_tensor = np.reshape(single_site_tensor, (1,1,dimension))
        tensor_list = [deepcopy(single_site_tensor[0,:,:])]
        tensor_list.extend([deepcopy(single_site_tensor) for _ in range(num_sites-2)])
        tensor_list.append(deepcopy(single_site_tensor[0,:,:]))
        return cls.from_tensor_list(tensor_list,
                                    node_prefix=node_prefix,
                                    root_site=root_site)

class MatrixProductOperator(MatrixProductTree, TTNO):
    pass
