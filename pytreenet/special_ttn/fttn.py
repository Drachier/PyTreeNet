"""
Provides the fork tree topology TTN.

A fork tree tensor network (FTTN) has the shape of a fork or comb. It consists
of a main chain of tensor nodes and to each a subchain is attached. For example::

     M0 -- S00 -- S01 -- S03
     |
     M1 -- S10 -- S11
     |
     M2 -- S20 -- S21
     |
     M3

Also contains the subclasses that combine the FTTN structure with the
properties of a TTNS or TTNO.
"""
from __future__ import annotations
from typing import Union

import numpy as np

from ..core.ttn import TreeTensorNetwork
from ..ttns import TreeTensorNetworkState
from ..core.node import Node
from ..ttno.ttno_class import TTNO
from ..util.ttn_exceptions import positivity_check

class ForkTreeTensorNetwork(TreeTensorNetwork):
    """
    An FTTN is a tree tensor network that consists of a main chain
     and subchains attached to each of the nodes in the main chain.
     Thus they look like a fork/comb::

     M0 -- S00 -- S01 -- S03
     |
     M1 -- S10 -- S11
     |
     M2 -- S20 -- S21
     |
     M3

    Attributes:
        main_identifier_prefix (str): A prefix set for all identifiers of nodes
            in the main chain. The identifiers will be `prefix + position` and
            used to automatically add nodes to the chain.
        subchain_identifier_prefix (str): A prefix set for all identifiers of
            nodes in a sub chain. The identifiers will be
                `prefix + mainchainposition + _ + subchainposition`
            and used to automatically add nodes to the chain.
        main_chain (List[ndarray]): A list of nodes in the main chain. The
            nodes are ordered according to the order in which they were added
            thus defining the order of the main chain.
        sub_chains (List[List[ndarray]]): A list of lists of nodes. Each list
            corresponds to a subchain attached to a node in the main chain.
            The nodes are ordered according to the order in which they were
            added thus defining the order of the subchain.
    """

    def __init__(self, main_identifier_prefix : str = "main",
                 subchain_identifier_prefix: str = "sub"):
        """
        Initilaises a ForkTreeTensorNetwork object.

        Note that the idenfitiers of the nodes are automatically generated and
        fixed.

        Args:
            main_identifier_prefix (str): Prefix of the main chain nodes.
                The identifiers will have the form `prefix + position in chain`.
                Defaults to "main".
            subchain_identifier_prefix (str): Prefix of the sub chain nodes.
                The identifiers will have the form `prefix + main chain node
                position + _ + subchain node position`. Defaults to "sub".
        """
        super().__init__()
        self.main_identifier_prefix = main_identifier_prefix
        self.subchain_identifier_prefix = subchain_identifier_prefix
        self.main_chain = []
        self.sub_chains = []

    def main_length(self) -> int:
        """
        Returns the number of nodes in the main chain.
        """
        return len(self.main_chain)

    def subchain_length(self, index: int) -> int:
        """
        Returns the number of tensors in the 'index'th subchain.
        """
        return len(self.sub_chains[index])

    def main_chain_id(self, index: int) -> str:
        """
        Creates the identifier of the 'index'th node in the main chain.
        """
        return self.main_identifier_prefix + str(index)

    def subchain_id(self, main_index: int, sub_index: int) -> str:
        """
        Creates the identifier of the node at the specified coordinates.

        Args:
            main_index (int): The index of the node according to the main chain.
            sub_index (int): The index of the node inside the subchain.

        Returns:
            str: The identifier of the node.
        """
        coord_str = str(main_index) + "_" + str(sub_index)
        return self.subchain_identifier_prefix + coord_str

    def add_main_chain_node(self, tensor: np.ndarray,
                            parent_leg: Union[None, int] = None):
        """
        Add a new node to the main chain.
        
        The identifier will be created automatically.

        Args:
            tensor (np.ndarray): The tensor that is to be associated to the
                new node.
            parent_leg (Union[None, int], optional): If there is already a node
                in the chain, the leg that should be attached to the new node
                can be specified. Otherwise it defaults to the first open leg.
        """
        main_length = self.main_length()
        identifier = self.main_chain_id(main_length)
        node = Node(identifier=identifier)
        if main_length== 0:
            self.add_root(node, tensor)
        else:
            parent_id = self.main_chain_id(main_length - 1)
            if parent_leg is None:
                parent_node = self.main_chain[-1]
                parent_leg = parent_node.nvirt_legs()
            self.add_child_to_parent(node, tensor, 0, parent_id, parent_leg)
        self.main_chain.append(node)
        self.sub_chains.append([])

    def add_sub_chain_node(self,
                           tensor: np.ndarray,
                           subchain_index: int,
                           parent_leg: Union[None, int] = None):
        """
        Add a new node to a given subchain.
        
        The identifier will be created automatically.

        Args:
            tensor (np.ndarray): The tensor that is to be associated to the new
                node.
            subchain_index (int): The index of the subchain to which the new
                node is to be added.
            parent_leg (Union[None, int], optional): If there is already a node
                in the chain, the leg that should be attached to the new node
                can be specified. Otherwise it defaults to the first open leg
                of the to be parent node.
        """
        if subchain_index > self.main_length():
            errstr = "A subchain has to be attached to the main chain!"
            raise ValueError(errstr)
        subchain_length = self.subchain_length(subchain_index)
        identifier = self.subchain_id(subchain_index, subchain_length)
        node = Node(identifier=identifier)
        if subchain_length == 0:
            parent_id = self.main_chain_id(subchain_index)
            if parent_leg is None:
                parent_node = self.main_chain[subchain_index]
                parent_leg = parent_node.nvirt_legs()
        else:
            parent_id = self.subchain_id(subchain_index, subchain_length - 1)
            if parent_leg is None:
                parent_node = self.sub_chains[subchain_index][-1]
                parent_leg = parent_node.nvirt_legs()
        self.add_child_to_parent(node, tensor, 0, parent_id, parent_leg)
        self.sub_chains[subchain_index].append(node)

class ForkTreeProductState(ForkTreeTensorNetwork, TreeTensorNetworkState):
    """
    A fork tree tensor network state.

    A fork tree tensor network state is a tensor network state that has the
    structure of a fork tree tensor network. It consists of a main chain of
    tensor nodes and to each a subchain is attached. For example::

         M0 -- S00 -- S01 -- S03
         |
         M1 -- S10 -- S11
         |
         M2 -- S20 -- S21
         |
         M3
    
    Every node has one open leg, that may be trivial.
    """
    def __init__(self,
                 main_identifier_prefix: str = "main",
                 subchain_identifier_prefix: str = "sub"):
        super().__init__(main_identifier_prefix,
                         subchain_identifier_prefix)

def constant_ftps(local_state: np.ndarray,
                  width: int,
                  height: int,
                  bond_dim: int = 1,
                  main_identifier_prefix: str = "main",
                  subchain_identifier_prefix: str = "sub"
                  ) -> ForkTreeProductState:
    """
    Generates an FTPS that has the same tensor at every node.

    Args:
        local_tensor (np.ndarray): The tensor that is to be used at every node.
        width (int): The number of nodes in the main chain.
        height (int): The number of nodes in each subchain.
        bond_dim (int, optional): The bond dimension to which the tensors are
            extended. Defaults to 1.

    Returns:
        ForkTreeProductState: The generated FTPS.
    
    """
    positivity_check(width, "width")
    positivity_check(height, "height")
    positivity_check(bond_dim, "bond_dim")
    assert local_state.ndim == 1, "The local state has to be a vector!"
    phys_dim = local_state.shape[0]
    ftps = ForkTreeProductState(main_identifier_prefix=main_identifier_prefix,
                                subchain_identifier_prefix=subchain_identifier_prefix)
    # Add main chains
    for i in range(height):
        if i in [0, height - 1]:
            tensor = np.zeros((bond_dim, bond_dim, phys_dim),
                              dtype=local_state.dtype)
            tensor[0, 0, :] = local_state
        else:
            tensor = np.zeros((bond_dim, bond_dim, bond_dim, phys_dim),
                              dtype=local_state.dtype)
            tensor[0, 0, 0, :] = local_state
        ftps.add_main_chain_node(tensor)
    # Add subchains
    for i in range(height):
        for j in range(width - 1):
            if j == width - 2:
                tensor = np.zeros((bond_dim, phys_dim),
                                  dtype=local_state.dtype)
                tensor[0, :] = local_state
            else:
                tensor = np.zeros((bond_dim, bond_dim, phys_dim),
                                  dtype=local_state.dtype)
                tensor[0, 0, :] = local_state
            ftps.add_sub_chain_node(tensor, i)
    return ftps

class ForkTreeProductOperator(ForkTreeTensorNetwork, TTNO):
    """
    A fork tree tensor network operator.

    A fork tree tensor network operator is a tensor network operator that has
    the structure of a fork tree tensor network. It consists of a main chain of
    tensor nodes and to each a subchain is attached. For example::

         M0 -- S00 -- S01 -- S03
         |
         M1 -- S10 -- S11
         |
         M2 -- S20 -- S21
         |
         M3
    
    Every node has two open legs, that may be trivial.
    """
    def __init__(self, main_identifier_prefix: str = "main", subchain_identifier_prefix: str = "sub"):
        super().__init__(main_identifier_prefix, subchain_identifier_prefix)
