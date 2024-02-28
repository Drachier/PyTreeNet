"""
This module provides functions to contract two TreeTensorNetworkStates.
"""

from __future__ import annotations
from typing import Union

import numpy as np

from .tree_cach_dict import PartialTreeCachDict
from ..node import Node

from .contraction_util import determine_leg_with_ignored_leg

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
    Contracts a node with its environment, assuming all subtrees going aeays
     from the node are already contracted.

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
            subtrees going away from the node.
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
    ketblock_tensor = contract_all_neighbour_blocks_to_ket(ket_tensor,
                                                           ket_node,
                                                           dictionary)
    bra_node, bra_tensor = state2[node_id]
    return contract_bra_to_ket_and_blocks(bra_tensor, ketblock_tensor,
                                          bra_node, ket_node)

def contract_any(node_id: str, next_node_id: str,
                 state1: TreeTensorNetworkState,
                 state2: TreeTensorNetworkState,
                 dictionary: PartialTreeCachDict) -> np.ndarray:
    node = state1.nodes[node_id]
    if node.is_leaf():
        return contract_leafs(node_id, state1, state2)
    return contract_subtrees_using_dictionary(node_id,
                                              next_node_id,
                                              state1,
                                              state2,
                                              dictionary)

def contract_leafs(node_id: str, state1: TreeTensorNetworkState,
                   state2: TreeTensorNetworkState) -> np.ndarray:
    """
    Creates a SubTreeSandwichContraction for a leaf node.

    Args:
        node_id (str): The identifier of the leaf node.
        state1 (TreeTensorNetworkState): The first TTN state.
        state2 (TreeTensorNetworkState): The second TTN state.

    Returns:
        np.ndarray: The tensor resulting from the contraction:
                     _____
                ____|     |
                    |  B  |
                    |_____|
                       |
                       |
                     __|__
                ____|     |
                    |  A  |
                    |_____|
            
            where B is the tensor in state2 and A is the tensor in state1.
    """
    node1 = state1.nodes[node_id]
    node2 = state2.nodes[node_id]
    assert node1.is_leaf() and node2.is_leaf()
    errstr = "The leaf nodes must have exactly one open leg."
    assert len(node1.open_legs) == 1 and len(node2.open_legs) == 1, errstr
    tensor = np.tensordot(state1.tensors[node_id], state2.tensors[node_id],
                            axes=(node1.open_legs[0], node2.open_legs[0]))
    return tensor

def contract_subtrees_using_dictionary(node_id: str, next_node_id: str,
                                       state1: TreeTensorNetworkState,
                                       state2: TreeTensorNetworkState,
                                       dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    The tensor with only two open legs, pointing to the same neighbour, after
     contracting a hole subtree. The subtrees attached to the other virtual
     legs are already contracted and stored in the dictionary.

    Args:
        node_id (str): The identifier of the node.
        next_node_id (str): The identifier of the node to which the remaining
         virtual legs point.
        state1 (TreeTensorNetworkState): The first TTN state.
        state2 (TreeTensorNetworkState): The second TTN state.
        dictionary (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
        
        Returns:
            np.ndarray: The resulting tensor. For example, if the nodes have
             two neighbours.

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
    ket_node = state1.nodes[node_id]
    ket_tensor = state1.tensors[node_id]
    ketblock_tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                                  ket_node,
                                                                  next_node_id,
                                                                  dictionary)
    bra_tensor = state2.tensors[node_id]
    bra_node = state2.nodes[node_id]
    return contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                         ketblock_tensor,
                                                         bra_node,
                                                         ket_node,
                                                         next_node_id)

def contract_neighbour_block_to_ket(ket_tensor: np.ndarray,
                                    ket_node: Node,
                                    neighbour_id: str,
                                    partial_tree_cache: PartialTreeCachDict,
                                    tensor_leg_to_neighbour: Union[None,int]=None) -> np.ndarray:
    """
    Contracts the ket tensor, i.e. A in the diagrams, with one neighbouring
     block, C in the diagrams.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        neighbour_id (str): The identifier of the neighbour node which is the
            root node of the subtree that has already been contracted and is
            saved in the dictionary.
        tensor_leg_to_neighbour (int): The index of the leg of the ket tensor
            that points to the neighbour block and is thus to be contracted.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
    
    Returns:
        np.ndarray: The resulting tensor.
                                    ______
                               ____|      |
                                   |      |
                                   |      |
                                   |      |
                           |       |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|
    """
    cached_neighbour_tensor = partial_tree_cache.get_entry(neighbour_id,
                                                           ket_node.identifier)
    if tensor_leg_to_neighbour is None:
        tensor_leg_to_neighbour = ket_node.neighbour_index(neighbour_id)
    return np.tensordot(ket_tensor, cached_neighbour_tensor,
                        axes=([tensor_leg_to_neighbour],[0]))

def contract_all_neighbour_blocks_to_ket(ket_tensor: np.ndarray,
                                         ket_node: Node,
                                         partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contract all neighbour blocks to the ket tensor.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.

    Returns:
        np.ndarray: The resulting tensor.
             ______                 ______
            |      |____       ____|      |
            |      |               |      |
            |      |               |      |
            |      |               |      |
            |  C1  |       |       |  C2  |
            |      |     __|__     |      |
            |      |____|     |____|      |
            |      |    |  A  |    |      |
            |______|    |_____|    |______|
        """
    result_tensor = ket_tensor
    for neighbour_id in ket_node.neighbouring_nodes():
        # A the neighbours are the same as the leg order, the tensor_leg_to_neighbour
        # is always 0.
        result_tensor = contract_neighbour_block_to_ket(result_tensor,
                                                        ket_node,
                                                        neighbour_id,
                                                        partial_tree_cache,
                                                        tensor_leg_to_neighbour=0)
    return result_tensor

def contract_neighbour_block_to_ket_ignore_one_leg(ket_tensor: np.ndarray,
                                                     ket_node: Node,
                                                     neighbour_id: str,
                                                     ignoring_node_id: str,
                                                     partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts the ket tensor, i.e. A in the diagrams, with one neighbouring
     block, C in the diagrams, ignoring one leg.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        neighbour_id (str): The identifier of the neighbour node which is the
            root node of the subtree that has already been contracted and is
            saved in the dictionary.
        ignoring_node_id (str): The identifier of the node to which the virtual
            leg should not point.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.

    Returns:
        np.ndarray: The resulting tensor.
                                    ______
                               ____|      |
                                   |      |
                                   |      |
                                   |      |
                           |       |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|
    """
    tensor_index_to_neighbour = determine_leg_with_ignored_leg(ket_node,
                                                               neighbour_id,
                                                               ignoring_node_id)
    return contract_neighbour_block_to_ket(ket_tensor, ket_node,
                                           neighbour_id,
                                           partial_tree_cache,
                                           tensor_leg_to_neighbour=tensor_index_to_neighbour)        

def contract_all_but_one_neighbour_block_to_ket(ket_tensor: np.ndarray,
                                                ket_node: Node,
                                                next_node_id: str,
                                                partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contract all neighbour blocks to the ket tensor.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
    """
    result_tensor = ket_tensor
    for neighbour_id in ket_node.neighbouring_nodes():
        if neighbour_id != next_node_id:
            result_tensor = contract_neighbour_block_to_ket_ignore_one_leg(result_tensor,
                                                                           ket_node,
                                                                           neighbour_id,
                                                                           next_node_id,
                                                                           partial_tree_cache)
    return result_tensor

def contract_bra_to_ket_and_blocks(bra_tensor: np.ndarray,
                                   ketblock_tensor: np.ndarray,
                                   bra_node: Node,
                                   ket_node: Node) -> np.ndarray:
    """
    Contracts the bra tensor, i.e. B in the diagrams, with the ketblock tensor,
        i.e. A and C in the diagrams.

    Args:
        bra_tensor (np.ndarray): The tensor of the bra node.
        ketblock_tensor (np.ndarray): The tensor resulting from the contraction
         of the ket node and its neighbouring blocks.
        bra_node (Node): The bra node.
        ket_node (Node): The ket node.

    Returns:
        np.ndarray: The resulting tensor.
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
                                                  next_node_id: str) -> np.ndarray:
    """
    Contracts the bra tensor, i.e. B in the diagrams, with the ketblock tensor,
     i.e. A and C in the diagrams.

    Args:
        bra_tensor (np.ndarray): The tensor of the bra node.
        ketblock_tensor (np.ndarray): The tensor resulting from the contraction
         of the ket node and its neighbouring blocks.
        bra_node (Node): The bra node.
        ket_node (Node): The ket node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.

    Returns:
        np.ndarray: The resulting tensor.

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
    for neighbour_id in bra_node.neighbouring_nodes():
        if neighbour_id != next_node_id:
            ket_index = ket_node.neighbour_index(neighbour_id)
            legs_block.append(ket_index + 1 + int(next_node_index > ket_index))
            legs_bra.append(bra_node.neighbour_index(neighbour_id))
    legs_block.append(1)
    num_neighbours = bra_node.nneighbours()
    legs_bra.append(num_neighbours)
    return np.tensordot(ketblock_tensor, bra_tensor, axes=(legs_block, legs_bra))
