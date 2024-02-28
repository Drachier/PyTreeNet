"""
This module provides functions to contract a TreeTensorNetworkOperator with
 a TreeTensorNetworkState.
"""

from __future__ import annotations

import numpy as np

from ..node import Node
from .tree_cach_dict import PartialTreeCachDict

from .contraction_util import (determine_index_with_ignored_leg,
                                get_equivalent_legs)

def for_leaf(node_id: str,
             state: TreeTensorNetworkState,
             hamiltonian: TTNO) -> np.ndarray:
    """
    If the current subtree ends and starts at a leaf, only the three
        tensors corresponding to that site must be contracted. Furthermore,
        the retained legs must point towards the leaf's parent.

    Args:
        node_id (str): Identifier of the leaf node
        state (TreeTensorNetworkState): The TTNS representing the state.
        hamiltonian (TTNO): The TTNO representing the Hamiltonian.

    Returns:
        np.ndarray: The contracted partial tree.
                     _____
           2    ____|     |
                    |  A* |
                    |_____|
                       |
                       |1
                     __|__
           1    ____|     |
                  0 |  H  |
                    |_____|
                       |2
                       |
                     __|__
           0    ____|     |
                    |  A  |
                    |_____|
    """
    ket_node, ket_tensor = state[node_id]
    bra_tensor = ket_tensor.conj()
    ham_node, ham_tensor = hamiltonian[node_id]
    bra_ham = np.tensordot(ham_tensor, bra_tensor,
                           axes=(_node_operator_output_leg(ham_node),
                                 _node_state_phys_leg(ket_node)))
    bra_ham_ket = np.tensordot(ket_tensor, bra_ham,
                               axes=(_node_state_phys_leg(ket_node),
                                     _node_operator_input_leg(ham_node)-1))
    return bra_ham_ket

def contract_neighbour_block_to_ket_ignore_one_leg(ket_tensor: np.ndarray,
                                                   ket_node: Node,
                                                   neighbour_id: str,
                                                   ignoring_node_id: str,
                                                   partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts a cached contracted tree C that originates at the neighbour
        node with the tensor of this node, which is currently a ket node A,
        possibly with other neighbour caches already contracted.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        neighbour_id (str): The identifier of the neighbour node which is the
         root node of the subtree that has already been contracted and is 
         saved in the dictionary.
        ignoring_node_id (str): The identifier of the node to which the
         virtual node should not point.
        partiral_tree_cache (PartialTreeCachDict): The dictionary containing
            the already contracted subtrees.
        
    Returns:
        np.ndarray: The contracted tensor:
                                ______
                               |      |
                        _______|      |
                               |      |
                               |      |
                               |      |
                               |      |
                        _______|      |
                               |  C   |
                               |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
                ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|
    """
    tensor_index_to_neighbour = determine_index_with_ignored_leg(ket_node,
                                                                 neighbour_id,
                                                                 ignoring_node_id)
    cached_neighbour_tensor = partial_tree_cache.get_entry(neighbour_id,
                                                           ket_node.identifier)
    return np.tensordot(ket_tensor, cached_neighbour_tensor,
                        axes=([tensor_index_to_neighbour],[0]))

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

def contract_operator_tensor_ignoring_one_leg(current_tensor: np.ndarray,
                                              ket_node: Node,
                                              op_tensor: np.ndarray,
                                              op_node: Node,
                                              ignoring_node_id: str) -> np.ndarray:
    """
    Contracts the operator tensor with the current tensor, which is a ket
     tensor to which the neighbour blocks were already contracted. One of
     the legs to a neighbour node is not contracted.
    
    Args:
        current_tensor (np.ndarray): The current tensor.
        ket_node (Node): The ket node.
        op_tensor (np.ndarray): The operator tensor.
        op_node (Node): The operator node.
        ignoring_node_id (str): The identifier of the node to which the
         virtual leg should not point.

    Returns:
        np.ndarray: The contracted tensor.
                                    ______
                                   |      |
                            _______|      |
                                   |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  H  |    |  C   |
                        |_____|    |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|
    """
    _, op_legs = get_equivalent_legs(ket_node, op_node, [ignoring_node_id])
    # Due to the legs to the bra tensor, the legs of the current tensor are a
    # bit more complicated
    tensor_legs = list(range(2,2*ket_node.nneighbours(),2))
    # Adding the physical legs
    tensor_legs.append(1)
    op_legs.append(_node_operator_output_leg(op_node))
    return np.tensordot(current_tensor, op_tensor,
                        axes=(tensor_legs, op_legs))

def _node_state_phys_leg(node: Node) -> int:
    """
    Finds the physical leg of a node of a state.

    Returns:
        int: The index of the physical leg.
    """
    return node.nneighbours()

def _node_operator_input_leg(node: Node) -> int:
    """
    Finds the leg of a node of the hamiltonian corresponding to the input.

    Returns:
        int: The index of the leg corresponding to input.
    """
    # Corr ket leg
    return node.nneighbours() + 1

def _node_operator_output_leg(node: Node) -> int:
    """
    Finds the leg of a node of the hamiltonian corresponding to the
        output.
    
    Returns:
        int: The index of the leg corresponding to output.
    """
    # Corr bra leg
    return node.nneighbours()
