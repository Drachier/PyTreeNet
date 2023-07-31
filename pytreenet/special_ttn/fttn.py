from __future__ import annotations
from typing import Union

import numpy as np

from ..ttn import TreeTensorNetwork
from ..node import Node

class ForkTreeTensorNetwork(TreeTensorNetwork):
    """
    An FTTN is a tree tensor network that consists of a main chain
     and subchains attached to each of the nodes in the main chain.
     Thus they look like a fork/comb:

     M0 -- S00 -- S01 -- S03
     |
     M1 -- S10
     |
     M2 -- S20 -- S21
     |
     M3
    """

    def __init__(self, main_identifier_prefix : str = "main",
                 subchain_identifier_prefix: str = "sub"):
        """
        Creates a FTTN. The identifiers of the nodes have a fixed pattern.
         Merely the prefixes identifying the main and subchain nodes can be
         changed.

        Args:
            main_identifier_prefix (str, optional): Prefix of the main chain nodes.
             The identifiers will have the form "prefix" + "position in chain".
             Defaults to "main".
            subchain_identifier_prefix (str, optional): Prefix of the sub chain nodes.
             The identifiers will have the form "prefix" + "main chain node position"
             + "_" + "subchain node position".
             Defaults to "sub".
        """
        super().__init__()
        self.main_identifier_prefix = main_identifier_prefix
        self.subchain_identifier_prefix = subchain_identifier_prefix
        self.main_chain = []
        self.sub_chains = []

    def main_length(self) -> int:
        """Returns the number of nodes in the main chain."""
        return len(self.main_chain)

    def subchain_length(self, index: int) -> int:
        """Returns the number of tensors in the 'index'th subchain."""
        return len(self.sub_chains[index])

    def add_main_chain_node(self, tensor: np.ndarray, parent_leg: Union[None, int] = None):
        """
        Add a new node to the main chain. The identifier will be created automatically.

        Args:
            tensor (np.ndarray): The tensor that is to be associated to the new node.
            parent_leg (Union[None, int], optional): If there is already a node in the chain,
             the leg that should be attached to the new node can be specified. Otherwise it
             defaults to the last virtual leg of the to be parent node.
        """
        main_length = self.main_length()
        identifier = self.main_identifier_prefix + str(main_length)
        node = Node(identifier=identifier)
        if main_length== 0:
            self.add_root(node, tensor)
        else:
            parent_id = self.main_identifier_prefix + str(main_length - 1)
            if parent_leg is None:
                parent_node = self.main_chain[-1]
                parent_leg = parent_node.nvirt_legs()
            self.add_child_to_parent(node, tensor, 0, parent_id, parent_leg)
        self.main_chain.append(node)
        self.sub_chains.append([])

    def add_sub_chain_node(self, tensor: np.ndarray, subchain_index: int,
                           parent_leg: Union[None, int] = None):
        """
        Add a new node to the sub chain with index 'subchain_index'.
         The identifier will be created automatically.

        Args:
            tensor (np.ndarray): The tensor that is to be associated to the new node.
            subchain_index (int): The index of the subchain to which the new node is to be added.
            parent_leg (Union[None, int], optional): If there is already a node in the chain,
             the leg that should be attached to the new node can be specified. Otherwise it
             defaults to the last virtual leg of the to be parent node.
        """
        if subchain_index > self.main_chain:
            errstr = "A subchain has to be attached to the main chain!"
            raise ValueError(errstr)
        subchain_length = self.subchain_length(subchain_index)
        identifier = self.subchain_identifier_prefix + str(subchain_index) + "_" + str(subchain_length)
        node = Node(identifier=identifier)
        if subchain_length == 0:
            parent_id = self.main_identifier_prefix + str(subchain_index)
            if parent_leg is None:
                parent_node = self.main_chain[subchain_index]
                parent_leg = parent_node.nvirt_legs()
        else:
            parent_id = self.subchain_identifier_prefix + str(subchain_index) + "_" + str(subchain_length - 1)
            if parent_leg is None:
                parent_node = self.sub_chains[subchain_index][-1]
                parent_leg = parent_node.nvirt_legs()
        self.add_child_to_parent(node, tensor, 0, parent_id, parent_leg)
        self.sub_chains[subchain_index].append(node)