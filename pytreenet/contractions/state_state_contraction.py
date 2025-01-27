"""
This module provides the function to contract two TreeTensorNetworkStates.

The main function is contract_two_ttns(ttn1, ttn2), which contracts two
TreeTensorNetworkStates with the same underlying tree structure.
Therefore the contraction is basically the scalar product of the first state
with the complex conjugate of the second state. <TTN2*|TTN1>

Using explicit imports from this module provides utility functions to contract
subtrees and leaf nodes of two states.
"""
from __future__ import annotations
from typing import Union, Callable

import numpy as np

from .tree_cach_dict import PartialTreeCachDict
from ..core.node import Node
from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                               contract_all_neighbour_blocks_to_ket) 

__all__ = ['contract_two_ttns']

def contract_two_ttns(ttn1: TreeTensorNetworkState,
                      ttn2: TreeTensorNetworkState) -> complex:
    """
    Contracts two TreeTensorNetworks.

    Args:
        ttn1 (TreeTensorNetwork): The first TreeTensorNetwork.
        ttn2 (TreeTensorNetwork): The second TreeTensorNetwork.

    Returns:
        complex: The resulting scalar product <TTN1|TTN2>
    """
    dictionary = PartialTreeCachDict()
    computation_order = ttn1.linearise() # Getting a linear list of all identifiers
    errstr = "The last element of the linearisation should be the root node."
    assert computation_order[-1] == ttn1.root_id, errstr
    assert computation_order[-1] == ttn2.root_id, errstr
    for node_id in computation_order[:-1]: # The last one is the root node
        node = ttn1.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any(node_id, parent_id,
                             ttn1, ttn2,
                             dictionary)
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    return complex(contract_node_with_environment(ttn1.root_id,
                                                  ttn1, ttn2,
                                                  dictionary))

def contract_node_with_environment(node_id: str,
                                   state1: TreeTensorNetworkState,
                                   state2: TreeTensorNetworkState,
                                   dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts a node with its environment.
     
    It is assumed that all subtrees starting from this node are already
    contracted.

    Args:
        node_id (str): The identifier of the node.
        state1 (TreeTensorNetworkState): The first TTN state.
        state2 (TreeTensorNetworkState): The second TTN state.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
    
    Returns:
        np.ndarray: The resulting tensor. A and B are the tensors in state1 and
            state2, respectively, corresponding to the node with the identifier
            node_id. C aer the tensors in the dictionary corresponding to the
            subtrees going away from the node::

             ______      _____      ______
            |      |____|     |____|      |
            |      |    |  B  |    |      |
            |      |    |_____|    |      |
            |      |       |       |      |
            |  C1  |       |       |  C2  |
            |      |     __|__     |      |
            |      |____|     |____|      |
            |      |    |  A  |    |      |
            |______|    |_____|    |______|
    
    """
    ket_node, ket_tensor = state1[node_id]
    bra_node, bra_tensor = state2[node_id]
    return contract_node_with_environment_nodes(ket_node, ket_tensor,
                                                bra_node, bra_tensor,
                                                dictionary)

def contract_node_with_environment_nodes(ket_node: Node,
                                         ket_tensor: np.ndarray,
                                         bra_node: Node,
                                         bra_tensor: np.ndarray,
                                         dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts a node with its environment.
     
    It is assumed that all subtrees starting from this node are already
    contracted.

    Args:
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        bra_node (Node): The bra node.
        bra_tensor (np.ndarray): The bra tensor.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
    
    Returns:
        np.ndarray: The resulting tensor. A and B are the tensors in state1 and
            state2, respectively, corresponding to the node with the identifier
            node_id. C aer the tensors in the dictionary corresponding to the
            subtrees going away from the node::

             ______      _____      ______
            |      |____|     |____|      |
            |      |    |  B  |    |      |
            |      |    |_____|    |      |
            |      |       |       |      |
            |  C1  |       |       |  C2  |
            |      |     __|__     |      |
            |      |____|     |____|      |
            |      |    |  A  |    |      |
            |______|    |_____|    |______|
    
    """
    ketblock_tensor = contract_all_neighbour_blocks_to_ket(ket_tensor,
                                                           ket_node,
                                                           dictionary)
    return contract_bra_to_ket_and_blocks(bra_tensor, ketblock_tensor,
                                          bra_node, ket_node)

def contract_any(node_id: str, next_node_id: str,
                 state1: TreeTensorNetworkState,
                 state2: TreeTensorNetworkState,
                 dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts any node.

    More specifically, it contracts the tensors of each state corresponding to
    the specified node with all but one of the subtrees attached to the node.
    The remaining open legs of the resulting tensor point to the uncontracted
    next node.

    Args:
        node_id (str): The identifier of the node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        state1 (TreeTensorNetworkState): The first TTN state.
        state2 (TreeTensorNetworkState): The second TTN state.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
    """
    node1, tensor1 = state1[node_id]
    node2, tensor2 = state2[node_id]
    return contract_any_nodes(next_node_id, node1, node2,
                              tensor1, tensor2, dictionary)

def contract_any_nodes(next_node_id: str,
                       node1: Node, node2: Node,
                       tensor1: np.ndarray, tensor2: np.ndarray,
                       dictionary: PartialTreeCachDict,
                       id_trafo: Union[Callable,None] = None) -> np.ndarray:
    """
    Contracts any two nodes using the given tensors.

    More specifically, it contracts the given tensors with all but one of the
    subtrees attached to the nodes. The remaining open legs of the resulting
    tensor point to the uncontracted next node.

    Note that the two nodes have to have the same connectivity, i.e. the same
    identifiers as neighbours.

    Args:
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        node1 (Node): The first node.
        node2 (Node): The second node.
        tensor1 (np.ndarray): The tensor corresponding to node1.
        tensor2 (np.ndarray): The tensor corresponding to node2.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        id_trafo (Union[Callable,None], optional): A function to transform the
            node identifiers of node1 into the node identifiers of node2. If
            None, it is assumed that the identifiers are the same. Defaults to
            None.
    
    Returns:
        np.ndarray: The resulting tensor. 

                 _____      ______
            ____|     |____|      |
            1   |  T2 |    |      |
                |_____|    |      |
                   |       |      |
                   |       |  C1  |
                 __|__     |      |
            ____|     |____|      |
            0   |  T1 |    |      |
                |_____|    |______|
    """
    if node1.is_leaf():
        return contract_leafs(node1, node2, tensor1, tensor2)
    return contract_subtrees_using_dictionary(next_node_id,
                                              node1, node2,
                                              tensor1, tensor2,
                                              dictionary,
                                              id_trafo=id_trafo)

def contract_leafs(node1: Node, node2: Node,
                   tensor1: np.ndarray, tensor2: np.ndarray
                   ) -> np.ndarray:
    """
    Contracts tensors associated with two leaf nodes.

    If the current subtree starts at a leaf node, only the tensors correspoding
    to this leaf in the two states must be contracted with each other.

    Args:
        node1 (Node): The first node.
        node2 (Node): The second node.
        tensor1 (np.ndarray): The tensor corresponding to node1.
        tensor2 (np.ndarray): The tensor corresponding to node2.

    Returns:
        np.ndarray: The tensor resulting from the contraction::

                     _____
                ____|     |
                    |  T2 |
                    |_____|
                       |
                       |
                     __|__
                ____|     |
                    |  T1 |
                    |_____|
    

    """
    assert node1.is_leaf() and node2.is_leaf()
    errstr = "The leaf nodes must have exactly one open leg."
    assert len(node1.open_legs) == 1 and len(node2.open_legs) == 1, errstr
    axes = (node1.open_legs[0], node2.open_legs[0])
    tensor = np.tensordot(tensor1, tensor2,
                            axes=axes)
    return tensor

def contract_subtrees_using_dictionary(next_node_id: str,
                                       node1: Node, node2: Node,
                                       tensor1: np.ndarray,
                                       tensor2: np.ndarray,
                                       dictionary: PartialTreeCachDict,
                                       id_trafo: Union[Callable,None] = None
                                       ) -> np.ndarray:
    """
    Contracts a node with all but one of the subtrees attached to it.

    The tensors corresponding to the nodes are contracted with
    each other and all but one subtrees starting from this node. The other
    subtrees are assumed to already be contracted and stored in the provided
    dictionary.

    Args:
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        node1 (Node): The first node.
        node2 (Node): The second node.
        tensor1 (np.ndarray): The tensor corresponding to node1.
        tensor2 (np.ndarray): The tensor corresponding to node2.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        id_trafo (Union[Callable,None], optional): A function to transform the
            node identifiers of node1 into the node identifiers of node2. If
            None, it is assumed that the identifiers are the same. Defaults to
            None.
        
        Returns:
            np.ndarray: The resulting tensor. For example, if the nodes have
                two neighbours::

                         _____      ______
                    ____|     |____|      |
                        |  T2 |    |      |
                        |_____|    |      |
                           |       |      |
                           |       |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  T1 |    |      |
                        |_____|    |______|
        
    """
    ket_node = node1
    ket_tensor = tensor1
    ketblock_tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                                  ket_node,
                                                                  next_node_id,
                                                                  dictionary)
    bra_node = node2
    bra_tensor = tensor2
    return contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                         ketblock_tensor,
                                                         bra_node,
                                                         ket_node,
                                                         next_node_id,
                                                         id_trafo=id_trafo)

def contract_bra_to_ket_and_blocks(bra_tensor: np.ndarray,
                                   ketblock_tensor: np.ndarray,
                                   bra_node: Node,
                                   ket_node: Node) -> np.ndarray:
    """
    Contracts the bra tensor with the ket and all neighbouring blocks.

    The bra tensor is contracted with the ket tensor and all neighbouring
    subtrees, which are already contracted into the blocks.

    Args:
        bra_tensor (np.ndarray): The tensor of the bra node.
        ketblock_tensor (np.ndarray): The tensor resulting from the contraction
         of the ket node and its neighbouring blocks.
        bra_node (Node): The bra node.
        ket_node (Node): The ket node.

    Returns:
        np.ndarray: The resulting tensor::
    
             ______      _____      ______
            |      |____|     |____|      |
            |      |    |  B  |    |      |
            |      |    |_____|    |      |
            |      |       |       |      |
            |  C1  |       |       |  C2  |
            |      |     __|__     |      |
            |      |____|     |____|      |
            |      |    |  A  |    |      |
            |______|    |_____|    |______|

    """
    num_neighbours = bra_node.nneighbours()
    legs_block = []
    for neighbour_id in bra_node.neighbouring_nodes():
        legs_block.append(ket_node.neighbour_index(neighbour_id) + 1)
    # The kets physical leg is now the first leg
    legs_block.append(0)
    # The bra tensor's physical leg is the last leg
    legs_bra = list(range(num_neighbours+1))
    return np.tensordot(ketblock_tensor, bra_tensor,
                        axes=(legs_block, legs_bra))

def contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor: np.ndarray,
                                                  ketblock_tensor: np.ndarray,
                                                  bra_node: Node,
                                                  ket_node: Node,
                                                  next_node_id: str,
                                                  id_trafo: Union[Callable,None] = None
                                                  ) -> np.ndarray:
    """
    Contracts the bra tensor with the ket and all but one neighbouring block.

    Args:
        bra_tensor (np.ndarray): The tensor of the bra node.
        ketblock_tensor (np.ndarray): The tensor resulting from the contraction
         of the ket node and its neighbouring blocks.
        bra_node (Node): The bra node.
        ket_node (Node): The ket node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        id_trafo (Union[Callable,None], optional): A function to transform the
            node identifiers of the ket node into the node identifiers of the
            the bra node. If None, it is assumed that the identifiers are the
            same. Defaults to None.

    Returns:
        np.ndarray: The resulting tensor::

                         _____      ______
                    ____|     |____|      |
                        |  B  |    |      |
                        |_____|    |      |
                           |       |      |
                           |       |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|
        
    """
    legs_block = []
    legs_bra = []
    next_node_index = ket_node.neighbour_index(next_node_id)
    for neighbour_id in ket_node.neighbouring_nodes():
        if neighbour_id != next_node_id:
            if id_trafo is None:
                bra_neighbour_id = neighbour_id
            else:
                bra_neighbour_id = id_trafo(neighbour_id)
            ket_index = ket_node.neighbour_index(neighbour_id)
            # 1 comes from the physical leg and the other one happens exactly
            # when the current neighbour index is above the ignored index.
            # This is how the block tensor is constructed.
            legs_block.append(ket_index + 1 + int(next_node_index > ket_index))
            bra_index = bra_node.neighbour_index(bra_neighbour_id)
            legs_bra.append(bra_index)
    # The physical leg of the ket is now leg 1
    legs_block.append(1)
    # Physical leg of the bra is the last leg
    num_neighbours = bra_node.nneighbours()
    legs_bra.append(num_neighbours)
    return np.tensordot(ketblock_tensor, bra_tensor,
                        axes=(legs_block, legs_bra))
