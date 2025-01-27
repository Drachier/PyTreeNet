"""
This module provides functions to contract a TTNS with a TTNO
"""

from __future__ import annotations
from typing import Union, Callable

import numpy as np

from ..core.node import Node
from .tree_cach_dict import PartialTreeCachDict
from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                               contract_all_neighbour_blocks_to_ket,
                               get_equivalent_legs)

__all__ = ['expectation_value', 'single_node_expectation_value']

def expectation_value(state: TreeTensorNetworkState,
                      operator: TTNO) -> complex:
    """
    Computes the Expecation value of a state with respect to an operator.

    The operator is given as a TTNO and the state as a TTNS. The expectation
    is obtained by "sandwiching" the operator between the state and its complex
    conjugate: <psi|H|psi>.

    Args:
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Operator.

    Returns:
        complex: The expectation value.
    """
    dictionary = PartialTreeCachDict()
    # Getting a linear list of all identifiers
    computation_order = state.linearise()
    errstr = "The last element of the linearisation should be the root node."
    assert computation_order[-1] == state.root_id, errstr
    assert computation_order[-1] == operator.root_id, errstr
    for node_id in computation_order[:-1]: # The last one is the root node
        node = state.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any(node_id, parent_id,
                             state, operator,
                             dictionary)
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    return complex(contract_node_with_environment(state.root_id,
                                                  state, operator,
                                                  dictionary))

def contract_node_with_environment(node_id: str,
                                   state: TreeTensorNetworkState,
                                   operator: TTNO,
                                   dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts a node with its environment.

    Assumes that all subtrees starting from this node are already contracted
    and the results stored in the dictionary.

    Args:
        node_id (str): The identifier of the node.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
    
    Returns:
        np.ndarray: The resulting tensor. A and B are the tensors in state1 and
            state2, respectively, corresponding to the node with the identifier
            node_id. C aer the tensors in the dictionary corresponding to the
            subtrees going away from the node::

                            ______
                 _____     |      |      _____
                |     |____|  A*  |_____|     |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|      |_____|     |
                |  C1 |    |   H  |     |  C2 |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|  A   |_____|     |
                |_____|    |______|     |_____|
    
    """
    ket_node, ket_tensor = state[node_id]
    ket_neigh_block = contract_all_neighbour_blocks_to_ket(ket_tensor,
                                                           ket_node,
                                                           dictionary)
    op_node, op_tensor = operator[node_id]
    state_legs, ham_legs = get_equivalent_legs(ket_node, op_node)
    ham_legs.append(_node_operator_input_leg(op_node))
    block_legs = list(range(1,2*ket_node.nneighbours(),2))
    block_legs.append(0)
    kethamblock = np.tensordot(ket_neigh_block, op_tensor,
                               axes=(block_legs, ham_legs))
    bra_tensor = ket_tensor.conj()
    state_legs.append(len(state_legs))
    return np.tensordot(bra_tensor, kethamblock,
                        axes=(state_legs,state_legs))

def contract_any(node_id: str, next_node_id: str,
                 state: TreeTensorNetworkState,
                 operator: TTNO,
                 dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts any node. 
    
    Rather the entire subtree starting from the node is contracted. The
    subtrees below the node already have to be contracted, except for the
    specified neighbour.
    This function combines the two options of contracting a leaf node or
    a general node using the dictionary in one function.
    
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
    state_node, state_tensor = state[node_id]
    operator_node, operator_tensor = operator[node_id]
    return contract_any_node_environment_but_one(next_node_id,
                                                 state_node, state_tensor,
                                                 operator_node, operator_tensor,
                                                 dictionary)

def contract_any_node_environment_but_one(ignored_node_id: str,
                                            ket_node: Node, ket_tensor: np.ndarray,
                                            op_node: Node, op_tensor: np.ndarray,
                                            dictionary: PartialTreeCachDict,
                                            bra_node: Union[Node,None] = None,
                                            bra_tensor: Union[Node,None] = None,
                                            id_trafo_op: Union[Callable,None] = None,
                                            id_trafo_bra: Union[Callable,None] = None
                                            ) -> np.ndarray:
    """
    Contracts a node with all its subtrees except for one.

    All subtrees except for one are already contracted and stored in the
    dictionary. The one that is not contracted is the one that the remaining
    legs point towards. The dictionary has to be according to the node
    identifiers of the ket node neighbours.

    Args:
        ignored_node_id (str): Identifier of the ket node to which the
            remaining legs should point.
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        op_node (Node): The operator node.
        op_tensor (np.ndarray): The operator tensor.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        bra_node (Union[Node,None], optional): The bra node. If None, the bra
            tensor is assumed to be the adjoint of the given ket tensor.
        bra_tensor (Union[Node,None], optional): The bra tensor. If None, the
            bra tensor is assumed to be the adjoint of the given ket tensor.
        id_trafo_op (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            operator node's neighbours. If None, the identity is assumed.
        id_trafo_bra (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            bra node's neighbours. If None, the identity is assumed.

    Returns:
        np.ndarray: The contracted tensor::

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
    if ket_node.is_leaf():
        return contract_leaf(ket_node, ket_tensor,
                             op_node, op_tensor,
                             bra_node, bra_tensor)
    return contract_subtrees_using_dictionary(ignored_node_id,
                                              ket_node, ket_tensor,
                                              op_node, op_tensor,
                                              dictionary,
                                              bra_node=bra_node, bra_tensor=bra_tensor,
                                              id_trafo_op=id_trafo_op,
                                              id_trafo_bra=id_trafo_bra)

def contract_leaf(state_node: Node,
                  state_tensor: np.ndarray,
                  operator_node: Node,
                  operator_tensor: np.ndarray,
                  bra_node: Union[None,Node] = None,
                  bra_tensor: Union[None,Node] = None) -> np.ndarray:
    """
    Contracts for a leaf node the state, operator and conjugate state tensors.

    If the current subtree starts at a leaf, only the three tensors
    corresponding to that site must be contracted. Furthermore, the retained
    legs must point towards the leaf's parent.

    Args:
        state_node (Node): The node of the state.
        state_tensor (np.ndarray): The tensor of the state.
        operator_node (Node): The node of the operator.
        operator_tensor (np.ndarray): The tensor of the operator.
        bra_node (Union[None,Node], optional): The node of the bra. If None,
            the bra tensor is assumed to be the adjoint of the given state
            tensor.
        bra_tensor (Union[None,Node], optional): The tensor of the bra. If None,
            the bra tensor is assumed to be the adjoint of the given state
            tensor.

    Returns:
        np.ndarray: The contracted partial tree::
    
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
    if bra_node is None and bra_tensor is None:
        bra_node = state_node
        bra_tensor = state_tensor.conj()
    elif bra_node is None or bra_tensor is None:
        errstr = "Either both bra_node and bra_tensor must be None or both must be given!"
        raise ValueError(errstr)
    bra_ham = np.tensordot(operator_tensor, bra_tensor,
                           axes=(_node_operator_output_leg(operator_node),
                                 _node_state_phys_leg(bra_node)))
    bra_ham_ket = np.tensordot(state_tensor, bra_ham,
                               axes=(_node_state_phys_leg(state_node),
                                     _node_operator_input_leg(operator_node)-1))
    return bra_ham_ket

def contract_subtrees_using_dictionary(ignored_node_id: str,
                                        ket_node: Node, ket_tensor: np.ndarray,
                                        op_node: Node, op_tensor: np.ndarray,
                                        dictionary: PartialTreeCachDict,
                                        bra_node: Union[Node,None] = None,
                                        bra_tensor: Union[Node,None] = None,
                                        id_trafo_op: Union[Callable,None] = None,
                                        id_trafo_bra: Union[Callable,None] = None) -> np.ndarray:
    """
    Contracts a node with all its subtrees except for one.

    All subtrees except for one are already contracted and stored in the
    dictionary. The one that is not contracted is the one that the remaining
    legs point towards.

    Args:
        ignored_node_id (str): Identifier of the ket node to which the
            remaining legs should point.
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        op_node (Node): The operator node.
        op_tensor (np.ndarray): The operator tensor.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        bra_node (Union[Node,None], optional): The bra node. If None, the bra
            tensor is assumed to be the adjoint of the given ket tensor.
        bra_tensor (Union[Node,None], optional): The bra tensor. If None, the
            bra tensor is assumed to be the adjoint of the given ket tensor.
        id_trafo_op (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            operator node's neighbours. If None, the identity is assumed.
        id_trafo_bra (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            bra node's neighbours. If None, the identity is assumed.

    Returns:
        np.ndarray: The contracted tensor::

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
    tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                         ket_node,
                                                         ignored_node_id,
                                                         dictionary)
    tensor = contract_operator_tensor_ignoring_one_leg(tensor,
                                                       ket_node,
                                                       op_tensor,
                                                       op_node,
                                                       ignored_node_id,
                                                       id_trafo=id_trafo_op)
    if bra_node is None and bra_tensor is None:
        bra_node = ket_node
        bra_tensor = ket_tensor.conj()
    elif bra_node is None or bra_tensor is None:
        errstr = "Either both bra_node and bra_tensor must be None or both must be given!"
        raise ValueError(errstr)
    environment_block = contract_bra_tensor_ignore_one_leg(bra_tensor,
                                                           bra_node,
                                                           tensor,
                                                           ket_node,
                                                           ignored_node_id,
                                                           id_trafo=id_trafo_bra)
    return environment_block

def contract_operator_tensor_ignoring_one_leg(current_tensor: np.ndarray,
                                              ket_node: Node,
                                              op_tensor: np.ndarray,
                                              op_node: Node,
                                              ignoring_node_id: str,
                                              id_trafo: Union[Callable,None] = None
                                              ) -> np.ndarray:
    """
    Contracts the operator tensor with the current tensor.

    The current tensor is the ket tensor of this node to which all but
    one neighbour blocks are already contracted. The blocks are the already
    contracted subtrees starting from this node. The subtree that is not
    contracted is the one that the remaining legs point towards.
    
    Args:
        current_tensor (np.ndarray): The current tensor.
        ket_node (Node): The ket node.
        op_tensor (np.ndarray): The operator tensor.
        op_node (Node): The operator node.
        ignoring_node_id (str): The identifier of the node to which the
            virtual leg should not point.
        id_trafo (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            operator node's neighbours. If None, the identity is assumed.

    Returns:
        np.ndarray: The contracted tensor::
    
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
    _, op_legs = get_equivalent_legs(ket_node, op_node,
                                     [ignoring_node_id],
                                     id_trafo=id_trafo)
    # Due to the legs to the bra tensor, the legs of the current tensor are a
    # bit more complicated
    tensor_legs = list(range(2,2*ket_node.nneighbours(),2))
    # Adding the physical legs
    tensor_legs.append(1)
    op_legs.append(_node_operator_input_leg(op_node))
    return np.tensordot(current_tensor, op_tensor,
                        axes=(tensor_legs, op_legs))

def contract_bra_tensor_ignore_one_leg(bra_tensor: np.ndarray,
                                       bra_node: Node,
                                       ketopblock_tensor: np.ndarray,
                                       ket_node: Node,
                                       ignoring_node_id: str,
                                       id_trafo: Union[Callable,None] = None
                                       ) -> np.ndarray:
    """
    Contracts the bra tensor with the contracted tensor.

    The current tensor has the ket tensor and the operator tensor of this
    node already contracted with each other and with all but one neighbour
    block. The remaining neighbour block is the one that the remaining legs
    point towards. The neighbour blocks are the results of already contracted
    subtrees starting from this node.

    Args:
        bra_tensor (np.ndarray): The bra tensor B.
        bra_node (Node): The bra node.
        ketopblock_tensor (np.ndarray): The contracted tensor. (ACH in the
            diagram)
        ket_node (Node): The node of the state. We assume the bra state
            is the adjoint of the ket state.
        ignoring_node_id (str): The identifier of the node to which the
            virtual leg should not point.
        id_trafo (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            bra node's neighbours. If None, the identity is assumed.

    Returns:
        np.ndarray: The contracted tensor::
                                    
                     _____      ______
                ____|     |____|      |
                    |  B  |    |      |
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
    num_neighbours = ket_node.nneighbours()
    legs_tensor = list(range(1,num_neighbours))
    legs_tensor.append(num_neighbours+1)
    _, legs_bra_tensor = get_equivalent_legs(ket_node, bra_node,
                                          [ignoring_node_id],
                                          id_trafo=id_trafo)
    # Adding the physical leg to be contracted.
    legs_bra_tensor.append(_node_state_phys_leg(bra_node))
    return np.tensordot(ketopblock_tensor, bra_tensor,
                        axes=(legs_tensor, legs_bra_tensor))

def single_node_expectation_value(node: Node,
                                  ket_tensor: np.ndarray,
                                  operator_tensor: np.ndarray,
                                  bra_tensor: Union[np.ndarray,None] = None
                                  ) -> complex:
    """
    This is a function for the special case, in which a TTN is only one node.

    The function computes the expectation value of the given ket tensor with
    respect to the given operator tensor.

    Args:
        node (Node): The node of the TTN.
        ket_tensor (np.ndarray): The ket tensor.
        operator_tensor (np.ndarray): The operator tensor.
        bra_tensor (Union[np.ndarray,None], optional): The bra tensor. If None,
            the bra tensor is assumed to be the adjoint of the given ket tensor.

    Returns:
        complex: The expectation value.

    """
    assert node.is_leaf() and node.is_root(), "Node is not a leaf and a root!"
    assert ket_tensor.ndim == 1, "Ket tensor has too many legs"
    assert operator_tensor.ndim == 2, "Operator tensor has too many legs"
    if bra_tensor is None:
        bra_tensor = ket_tensor.conj().T
    else:
        assert bra_tensor.ndim == 1, "Bra tensor has too many legs"
    return bra_tensor @ operator_tensor @ ket_tensor

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
    Finds the leg of a node of the hamiltonian corresponding to the output.
    
    Returns:
        int: The index of the leg corresponding to output.
    """
    # Corr bra leg
    return node.nneighbours()

def env_tensor_ket_leg_index():
    """
    Returns the index of the leg of the environment tensor corresponding to a
    ket tensors virtual leg
    
    """
    return 0

def env_tensor_ham_leg_index():
    """
    Returns the index of the leg of the environment tensor corresponding to a
    hamiltonian tensors virtual leg
    
    """
    return 1

def env_tensor_bra_leg_index():
    """
    Returns the index of the leg of the environment tensor corresponding to a
    bra tensors virtual leg
    
    """
    return 2
