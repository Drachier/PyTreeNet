"""
This module provides functions to contract a TreeTensorNetworkOperator with
 a TreeTensorNetworkState.
"""

from __future__ import annotations

import numpy as np

from ..core.node import Node
from .tree_cach_dict import PartialTreeCachDict

from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                                get_equivalent_legs)

def contract_any(node_id: str, next_node_id: str,
                 state: TreeTensorNetworkState,
                 operator: TTNO,
                 dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts any node, either as a leaf or using the dictionary with
     already contracted subtrees.
    
    Args:
        node_id (str): Identifier of the node.
        next_node_id (str): Identifier of the node towards which the open
         legs will point.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCachDict): The dictionary containing the
         already contracted subtrees.
        
    Returns:
        np.ndarray: The contracted tensor.
    """
    node = state.nodes[node_id]
    if node.is_leaf():
        return contract_leaf(node_id, state, operator)
    return contract_subtrees_using_dictionary(node_id,
                                              next_node_id,
                                              state,
                                              operator,
                                              dictionary)

def contract_leaf(node_id: str,
                  state: TreeTensorNetworkState,
                  operator: TTNO) -> np.ndarray:
    """
    If the current subtree ends and starts at a leaf, only the three
        tensors corresponding to that site must be contracted. Furthermore,
        the retained legs must point towards the leaf's parent.

    Args:
        node_id (str): Identifier of the leaf node
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.

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
    ham_node, ham_tensor = operator[node_id]
    bra_ham = np.tensordot(ham_tensor, bra_tensor,
                           axes=(_node_operator_output_leg(ham_node),
                                 _node_state_phys_leg(ket_node)))
    bra_ham_ket = np.tensordot(ket_tensor, bra_ham,
                               axes=(_node_state_phys_leg(ket_node),
                                     _node_operator_input_leg(ham_node)-1))
    return bra_ham_ket

def contract_subtrees_using_dictionary(node_id: str, ignored_node_id: str,
                                       state: TreeTensorNetworkState,
                                       operator: TTNO,
                                       dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contract all subtree blocks, i.e. already contracted subtrees, starting
     from neighbouring nodes, except for one. The one that is not contracted
     is the one that the remaining legs point towards.

    Args:
        node_id (str): Identifier of the node.
        ignored_node_id (str): Identifier of the node to which the remaining
         legs should point.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the operator.
        dictionary (PartialTreeCachDict): The dictionary containing the
         already contracted subtrees.

    Returns:
        np.ndarray: The contracted tensor:
                     _____      ______
              2 ____|     |____|      |
                    |  A* |    |      |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
              1 ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
              0 ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|
    """
    ket_node, ket_tensor = state[node_id]
    tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                         ket_node,
                                                         ignored_node_id,
                                                         dictionary)
    op_node, op_tensor = operator[node_id]
    tensor = contract_operator_tensor_ignoring_one_leg(tensor,
                                                       ket_node,
                                                       op_tensor,
                                                       op_node,
                                                       ignored_node_id)
    bra_tensor = ket_tensor.conj()
    return contract_bra_tensor_ignore_one_leg(bra_tensor,
                                              tensor,
                                              ket_node,
                                              ignored_node_id)

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
    op_legs.append(_node_operator_input_leg(op_node))
    return np.tensordot(current_tensor, op_tensor,
                        axes=(tensor_legs, op_legs))

def contract_bra_tensor_ignore_one_leg(bra_tensor: np.ndarray,
                                       ketopblock_tensor: np.ndarray,
                                       state_node: Node,
                                       ignoring_node_id: str) -> np.ndarray:
    """
    Contracts the bra tensor with the contracted tensor. The already
      contracted tensor has all the other neighbouring blocks of contracted
      subtrees contracted to it already.

    Args:
        bra_tensor (np.ndarray): The bra tensor.
        ketopblock_tensor (np.ndarray): The contracted tensor.
         (ACH in the diagram)
        state_node (Node): The node of the state. We assume the bra state
         is the adjoint of the ket state.
        ignoring_node_id (str): The identifier of the node to which the
            virtual leg should not point.

    Returns:
        np.ndarray: The contracted tensor.
                                    
                     _____      ______
                ____|     |____|      |
                    |  A* |    |      |
                    |_____|    |      |
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
    num_neighbours = state_node.nneighbours()
    ignored_node_index = state_node.neighbour_index(ignoring_node_id)
    legs_tensor = list(range(1,num_neighbours))
    legs_tensor.append(num_neighbours+1)
    legs_bra_tensor = list(range(ignored_node_index))
    legs_bra_tensor.extend(range(ignored_node_index+1,num_neighbours+1))
    return np.tensordot(ketopblock_tensor, bra_tensor,
                        axes=(legs_tensor, legs_bra_tensor))

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
