"""
This module provides functions to contract a TTNS with a TTNO
"""

from __future__ import annotations
from typing import Union, Callable, TYPE_CHECKING
from enum import Enum

import numpy as np

from .tree_cach_dict import PartialTreeCachDict
from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                               contract_all_neighbour_blocks_to_ket,
                               contract_all_but_one_neighbour_block_to_hamiltonian,
                               contract_all_neighbour_blocks_to_hamiltonian,
                               get_equivalent_legs)
from .state_state_contraction import contract_node_with_environment_nodes as env_contract_node_state

if TYPE_CHECKING:
    from ..core.node import Node
    from ..ttns.ttns import TreeTensorNetworkState
    from ..ttno.ttno_class import TreeTensorNetworkOperator

__all__ = ['expectation_value', 'single_node_expectation_value']

def expectation_value(state: TreeTensorNetworkState,
                      operator: TreeTensorNetworkOperator) -> complex:
    """
    Computes the Expecation value of a state with respect to an operator.

    The operator is given as a TTNO and the state as a TTNS. The expectation
    is obtained by "sandwiching" the operator between the state and its complex
    conjugate: <psi|H|psi>.

    Args:
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TreeTensorNetworkOperator): The TTNO representing the Operator.

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
    contr_order = compare_contr_orders(state.root[0], operator.root[0])
    # Compare which contraction is more efficient, first TTNS then TTNO
    # (contract_node_with_environment) or first TTNO then TTNS
    # (contract_node_with_environment_2).
    if state.root[0].nneighbours() > 1 and contr_order is FirstContraction.OPERATOR:
    # if 0:
        result = contract_node_with_environment_2(state.root_id,
                                                  state, operator,
                                                  dictionary)
    else:
        result = contract_node_with_environment(state.root_id,
                                                  state, operator,
                                                  dictionary)
    return complex(result)

def get_matrix_element(bra_state: TreeTensorNetworkState, 
                       operator: TreeTensorNetworkOperator,
                       ket_state: TreeTensorNetworkState) -> complex:
    """
    Computes the matrix element of an operator between a bra and a ket state.

    The operator is given as a TTNO and the state as a TTNS. The matrix
    element is obtained by "sandwiching" the operator between the bra and
    ket state: <bra|H|state>.

    Args:
        bra_state (TreeTensorNetworkState): The TTNS representing the bra state.
        operator (TreeTensorNetworkOperator): The TTNO representing the Operator.
        ket_state (TreeTensorNetworkState): The TTNS representing the ket state.

    Returns:
        complex: The expectation value.
    """
    dictionary = PartialTreeCachDict()
    # Getting a linear list of all identifiers
    computation_order = ket_state.linearise()
    errstr = "The last element of the linearisation should be the root node."
    assert computation_order[-1] == ket_state.root_id, errstr
    assert computation_order[-1] == operator.root_id, errstr
    for node_id in computation_order[:-1]: # The last one is the root node
        node = ket_state.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any(node_id, parent_id,
                             ket_state, operator,
                             dictionary,
                             bra_state=bra_state)
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    contr_order = compare_contr_orders(ket_state.root[0], operator.root[0])
    # Compare which contraction is more efficient, first TTNS then TTNO
    # (contract_node_with_environment) or first TTNO then TTNS
    # (contract_node_with_environment_2).
    if ket_state.root[0].nneighbours() > 1 and contr_order is FirstContraction.OPERATOR:
    # if 0:
        result = contract_node_with_environment_2(ket_state.root_id,
                                                  ket_state, operator,
                                                  dictionary,
                                                  bra_state=bra_state)
    else:
        result = contract_node_with_environment(ket_state.root_id,
                                                  ket_state, operator,
                                                  dictionary,
                                                  bra_state=bra_state)
    return complex(result)

class FirstContraction(Enum):
    """
    An enum to decide which node is contracted to the environments first.

    Attributes
        KET (str): Contract the ket node first.
        OPERATOR (str): Contract the operator node first.
    """
    KET = "ket"
    OPERATOR = "operator"

def compare_contr_orders(ket_node: Node,
                         op_node: Node
                         ) -> FirstContraction:
    """
    Compares the contraction cost of different orders of contracting.

    Args:
        ket_node (Node): The ket node.
        op_node (Node): The operator node.
    
    Returns:
        FirstContraction: The preferred contraction order.
    """
    phys_dim = ket_node.open_dimension()
    op_neigh_dim = op_node.virtual_dimension()
    ket_neigh_dim = ket_node.virtual_dimension()
    cost_lhs = (phys_dim - 1) * op_neigh_dim * ket_neigh_dim
    cost_rhs = phys_dim * (op_neigh_dim -  ket_neigh_dim)
    if cost_lhs >= cost_rhs:
        return FirstContraction.KET
    return FirstContraction.OPERATOR

def contract_node_with_environment(node_id: str,
                                   state: TreeTensorNetworkState,
                                   operator: TreeTensorNetworkOperator,
                                   dictionary: PartialTreeCachDict,
                                   bra_state: Union[TreeTensorNetworkState,None] = None
                                   ) -> np.ndarray:
    """
    Contracts a node with its environment.

    Assumes that all subtrees starting from this node are already contracted
    and the results stored in the dictionary.

    Args:
        node_id (str): The identifier of the node.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TreeTensorNetworkOperator): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
        bra_state (Union[TreeTensorNetworkState,None]): The TTNS representing the bra state.

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
    op_node, op_tensor = operator[node_id]
    kethamblock = contract_ket_ham_with_envs(ket_node,
                                             ket_tensor,
                                             op_node,
                                             op_tensor,
                                             dictionary)
    # TODO: Getting legs can also be done smarter, but works well enough for now.
    state_legs = list(range(ket_node.nlegs()))

    if bra_state is None:
        bra_tensor = ket_tensor.conj()
        return np.tensordot(bra_tensor, kethamblock,
                        axes=(state_legs,state_legs))
    bra_node, bra_tensor = bra_state[node_id]
    _, bra_legs = get_equivalent_legs(ket_node, bra_node)
    bra_legs.extend(bra_node.open_legs)
    return np.tensordot(bra_tensor, kethamblock,
                        axes=(bra_legs,state_legs))

def contract_ket_ham_with_envs(ket_node: Node,
                              ket_tensor: np.ndarray,
                              ham_node: Node,
                              ham_tensor: np.ndarray,
                              dictionary: PartialTreeCachDict
                              ) -> np.ndarray:
    """
    Contract a state node and a Hamiltonian node with their environments.

    Args:
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        ham_node (Node): The Hamiltonian node.
        ham_tensor (np.ndarray): The Hamiltonian tensor.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.

    Returns:
        np.ndarray: The contracted tensor.

                 _____                   _____
                |     |____        _____|     |
                |     |                 |     |
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
    ket_neigh_block = contract_all_neighbour_blocks_to_ket(ket_tensor,
                                                           ket_node,
                                                           dictionary)
    _, ham_legs = get_equivalent_legs(ket_node, ham_node)
    ham_legs.extend(_node_operator_input_leg(ham_node))
    nopen_ket = ket_node.nopen_legs()
    block_legs = list(range(nopen_ket,2*ket_node.nneighbours() + nopen_ket,2))
    block_legs.extend(list(range(nopen_ket)))
    kethamblock = np.tensordot(ket_neigh_block, ham_tensor,
                               axes=(block_legs, ham_legs))
    return kethamblock

def contract_node_with_environment_2(node_id: str,
                                   state: TreeTensorNetworkState,
                                   operator: TreeTensorNetworkOperator,
                                   dictionary: PartialTreeCachDict,
                                   bra_state: Union[TreeTensorNetworkState,None] = None
                                   ) -> np.ndarray:
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
        bra_state (Union[TreeTensorNetworkState,None]): The TTNS representing
         the bra state. If not given, the bra state is assumed to be the
         conjugate of the ket state.

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
    op_node, op_tensor = operator[node_id]
    op_neigh_block = contract_all_neighbour_blocks_to_hamiltonian(op_tensor,
                                                                   op_node,
                                                                   dictionary)

    _, ket_legs = get_equivalent_legs(op_node, ket_node)
    ket_legs.extend(ket_node.open_legs)
    nopen_op = op_node.nopen_legs()
    block_legs = list(range(nopen_op,2*op_node.nneighbours()+nopen_op,2))
    block_contr_open_legs = list(range(nopen_op // 2))
    block_legs.extend(block_contr_open_legs)
    kethamblock = np.tensordot(op_neigh_block, ket_tensor,
                               axes=(block_legs, ket_legs))
    if bra_state is None:
        bra_tensor = ket_tensor.conj()
        bra_legs = ket_legs
    else:
        bra_node, bra_tensor = bra_state[node_id]
        _, bra_legs = get_equivalent_legs(op_node, bra_node)
        bra_legs.extend(bra_node.open_legs)

    nopen_ket = ket_node.nopen_legs()
    block_legs = list(range(nopen_ket,op_node.nneighbours()+nopen_ket))+list(range(nopen_ket))

    result = np.tensordot(bra_tensor, kethamblock,
                        axes=(bra_legs,block_legs))
    return result

def contract_single_site_operator_env(ket_node: Node,
                                      ket_tensor: np.ndarray,
                                      bra_node: Node,
                                      bra_tensor: np.ndarray,
                                      operator: np.ndarray,
                                      dictionary: PartialTreeCachDict,
                                      id_trafo: Union[Callable,None] = None
                                      ) -> complex:
    """
    Contract a node to which a single-site operator is applied with given
    environment tensors.

    Args:
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        bra_node (Node): The bra node.
        bra_tensor (np.ndarray): The bra tensor.
        operator (np.ndarray): The single-site operator.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        id_trafo (Union[None,Callable]): A function to transform the node
            identifiers of the ket node into the node identifiers of the bra
            node. If None, it is assumed that the identifiers are the same.
            Defaults to None.

    Returns:
        complex: The contraction result.
                            ______
                 _____     |      |      _____
                |     |____|  B   |_____|     |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |    |      |     |     |
                |  C1 |    |  O   |     |  C2 |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|  K   |_____|     |
                |_____|    |______|     |_____|
    
    """
    # Contract the operator with the ket tensor
    # The ket's open leg is the last leg and the operators
    # input leg as well.
    ket_leg = ket_node.open_legs
    if len(ket_leg) != 1:
        raise ValueError("The ket node must have exactly one open leg!")
    ket_leg = ket_leg[0]
    new_ket_tensor = np.tensordot(ket_tensor, operator,
                                  axes=(ket_leg,1))
    return env_contract_node_state(ket_node, new_ket_tensor,
                                   bra_node, bra_tensor,
                                   dictionary,
                                   id_trafo=id_trafo)

def contract_any(node_id: str, next_node_id: str,
                 state: TreeTensorNetworkState,
                 operator: TreeTensorNetworkOperator,
                 dictionary: PartialTreeCachDict,
                 bra_state: Union[TreeTensorNetworkState,None] = None,
                 bra_state_conjugated: bool = True,
                 id_trafo_op: Union[Callable,None] = None,
                 id_trafo_bra: Union[Callable,None] = None
                    ) -> np.ndarray:
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
        operator (TreeTensorNetworkOperator): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        bra_state (Union[TreeTensorNetworkState,None]): The TTNS representing
            the bra state.
        bra_state_conjugated (bool): Whether the bra state is conjugated.
        id_trafo_op (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            operator node's neighbours. If None, the identity is assumed.
        id_trafo_bra (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            bra node's neighbours. If None, the identity is assumed.

    Returns:
        np.ndarray: The contracted tensor.
    """
    state_node, state_tensor = state[node_id]
    operator_node, operator_tensor = operator[node_id]
    if bra_state is None:
        bra_node, bra_tensor = state_node, state_tensor.conj()
    else:
        bra_node, bra_tensor = bra_state[node_id]
    if not bra_state_conjugated:
        bra_tensor = bra_tensor.conj()
    return contract_any_node_environment_but_one(next_node_id,
                                                 state_node, state_tensor,
                                                 operator_node, operator_tensor,
                                                 dictionary,
                                                 bra_node=bra_node,
                                                 bra_tensor=bra_tensor,
                                                 id_trafo_op=id_trafo_op,
                                                 id_trafo_bra=id_trafo_bra)

def contract_any_node_environment_but_one(ignored_node_id: str,
                                            ket_node: Node, ket_tensor: np.ndarray,
                                            op_node: Node, op_tensor: np.ndarray,
                                            dictionary: PartialTreeCachDict,
                                            bra_node: Union[Node,None] = None,
                                            bra_tensor: Union[np.ndarray,None] = None,
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
                  bra_tensor: Union[None,np.ndarray] = None) -> np.ndarray:
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
                    |  B  |
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
    input_legs = _node_operator_input_leg(operator_node)
    braham_legs = [index - len(input_legs) for index in input_legs]
    bra_ham_ket = np.tensordot(state_tensor, bra_ham,
                               axes=(_node_state_phys_leg(state_node),
                                     braham_legs))
    return bra_ham_ket

def contract_subtrees_using_dictionary(ignored_node_id: str,
                                        ket_node: Node, ket_tensor: np.ndarray,
                                        op_node: Node, op_tensor: np.ndarray,
                                        dictionary: PartialTreeCachDict,
                                        bra_node: Union[Node,None] = None,
                                        bra_tensor: Union[np.ndarray,None] = None,
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
    if bra_node is None and bra_tensor is None:
        bra_node = ket_node
        bra_tensor = ket_tensor.conj()
    elif bra_node is None or bra_tensor is None:
        errstr = "Either both bra_node and bra_tensor must be None or both must be given!"
        raise ValueError(errstr)
    contraction_order = compare_contr_orders(ket_node, op_node)
    if ket_node.nneighbours() > 1 and contraction_order is FirstContraction.OPERATOR:
        if id_trafo_op is None:
            id_trafo_op = lambda x: x
        tensor = contract_all_but_one_neighbour_block_to_hamiltonian(op_tensor,
                                                                     op_node,
                                                                     id_trafo_op(ignored_node_id),
                                                                     dictionary)
        tensor = contract_ket_tensor_ignoring_one_leg(tensor,
                                                      op_node,
                                                      ket_tensor,
                                                      ket_node,
                                                      ignored_node_id,
                                                      id_trafo=id_trafo_op)
    else:
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
    environment_block = contract_bra_tensor_ignore_one_leg(bra_tensor,
                                                            bra_node,
                                                            tensor,
                                                            ket_node,
                                                            ignored_node_id,
                                                            id_trafo=id_trafo_bra)
    return environment_block


def contract_ket_ham_ignoring_one_leg(ket_tensor: np.ndarray,
                                      ket_node: Node,
                                      op_tensor: np.ndarray,
                                      op_node: Node,
                                      ignored_node_id: str,
                                      dictionary: PartialTreeCachDict,
                                      id_trafo_op: Union[Callable,None] = None
                                      ) -> np.ndarray:
    """
    Contracts the ket tensor with the operator tensor and all but one neighbour
    block.

    Args:
        ket_tensor (np.ndarray): The ket tensor.
        ket_node (Node): The ket node.
        op_tensor (np.ndarray): The operator tensor.
        op_node (Node): The operator node.
        ignored_node_id (str): The identifier of the node to which the
            virtual leg should not point.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        id_trafo_op (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            operator node's neighbours. If None, the identity is assumed.

    Returns:
        np.ndarray: The contracted tensor. The ket tensor and the operator
            tensor are contracted with each other and all but one neighbour
            block.

                                    ______
                           1 to nn |      |
                            _______|      |
                                   |      |
                      nn+2 |       |      |
                           |       |      |
                    nn+1__ |__     |      |
                    ____|     |____|      |
                        |  H  |    |  C   |
                        |_____|    |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                    0   |  A  |    |      |
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
    return tensor

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
    num_ignored = 1
    _, op_legs = get_equivalent_legs(ket_node, op_node,
                                     [ignoring_node_id],
                                     id_trafo=id_trafo)
    op_legs.extend(_node_operator_input_leg(op_node))
    # Due to the legs to the bra tensor, the legs of the current tensor are a
    # bit more complicated
    ket_nopen = ket_node.nopen_legs()
    tensor_legs = list(range(ket_nopen + num_ignored,current_tensor.ndim,2))
    # Adding the physical legs
    block_open_legs = [index + num_ignored for index in range(ket_nopen)]
    tensor_legs.extend(block_open_legs)
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
    phys_indices = list(range(num_neighbours+1,
                              num_neighbours+1+bra_node.nopen_legs()))
    legs_tensor.extend(phys_indices)
    _, legs_bra_tensor = get_equivalent_legs(ket_node, bra_node,
                                          [ignoring_node_id],
                                          id_trafo=id_trafo)
    # Adding the physical leg to be contracted.
    legs_bra_tensor.extend(_node_state_phys_leg(bra_node))
    return np.tensordot(ketopblock_tensor, bra_tensor,
                        axes=(legs_tensor, legs_bra_tensor))

def contract_ket_tensor_ignoring_one_leg(current_tensor: np.ndarray,
                                              op_node: Node,
                                              ket_tensor: np.ndarray,
                                              ket_node: Node,
                                              ignoring_node_id: str,
                                              id_trafo: Union[Callable,None] = None
                                              ) -> np.ndarray:
    """
    Contracts the ket tensor with the current operator block tensor.

    The current tensor is the operator block tensor of this node to which all but
    one neighbour blocks are already contracted. The blocks are the already
    contracted subtrees starting from this node. The subtree that is not
    contracted is the one that the remaining legs point towards.
    
    Args:
        current_tensor (np.ndarray): The current operator block tensor.
        op_node (Node): The operator node.
        ket_tensor (np.ndarray): The ket tensor.
        ket_node (Node): The ket node.
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
                            1,2,3..|      |
                           |       |      |
              nneighbour+1 |       |      |
                         __|__     |      |
                    ____|     |____|      |
            nneighbour  |  H  |    |  C   |
                        |_____|    |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                     0  |  A  |    |      |
                        |_____|    |______|
    
    
    """
    if id_trafo is None:
        id_trafo = lambda x: x
    ket_legs, op_legs = get_equivalent_legs(ket_node, op_node,
                                     [ignoring_node_id],
                                     id_trafo=id_trafo)
    # The  legs of the block tensor are more complicated
    offset = op_node.nopen_legs() + 1 # One leg ignored
    ign_neighbour_index = op_node.neighbour_index(id_trafo(ignoring_node_id))
    tensor_legs = [(i - int(i > ign_neighbour_index)) * 2 + offset
                   for i in op_legs]
    # Adding the physical legs
    ket_open = ket_node.nopen_legs()
    tensor_legs.extend(range(ket_open+1, 2*ket_open+1))
    ket_legs.extend(ket_node.open_legs)
    result= np.tensordot(ket_tensor, current_tensor,
                        axes=(ket_legs, tensor_legs))
    permutation = [0]
    permutation += list(range(2+ket_open, result.ndim)) # The block legs go to the front
    permutation += list(range(1, ket_open + 2)) # The physical leg and the ignored leg go to the back
    result = np.transpose(result, permutation)
    return result

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

def contract_bond_tensor(
                        ket_tensor: np.ndarray,
                        ket_node: Node,
                        tensor_cache: PartialTreeCachDict
                        ) -> np.ndarray:
    """
    Contract an effective bond tensor with its neighbouring blocks.

    Args:
        state_tensor (NDArray[np.complex128]): The tensor of the state to be
            updated, representing a bond in the TTNS without an equivalent
            tensor in the TTNO.
        state_node (Node): The node of the state tensor.
        tensor_cache (SandwichCache): The cache for the neighbour blocks.

    Returns:
        NDArray[np.complex128]: The contracted tensor. The ket tensor and the
            neighbouring blocks are contracted with each other.

                 _____                   _____
                |     |____        _____|     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |_________________|     |
                |  C1 |                 |  C2 |
                |     |                 |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|  A   |_____|     |
                |_____|    |______|     |_____|
    """
    if ket_node.nneighbours()!= 2:
        errstr = "The node must have exactly two neighbours to perform " \
                "effective bond time evolution!"
        errstr += f" Node {ket_node.identifier} has {ket_node.nneighbours()} neighbours."
        raise ValueError(errstr)
    # We want to ensure the order of the legs
    order = [ket_node.parent,ket_node.children[0]]
    # The environments in the cache point to the other neighbour
    # Thus, we have to add keys to the cache
    ket_id = ket_node.identifier
    tensor_cache.add_entry(order[0], ket_id,
                           tensor_cache.get_entry(order[0], order[1]))
    tensor_cache.add_entry(order[1], ket_id,
                            tensor_cache.get_entry(order[1], order[0]))
    ketandblocks = contract_all_neighbour_blocks_to_ket(ket_tensor,
                                                        ket_node,
                                                        tensor_cache,
                                                        order=order)
    # Delete the new keys from the cache again
    for node_id in order:
        tensor_cache.delete_entry(node_id, ket_id)
    # Now we connect the Hamiltonian legs of the two neighbours.
    return np.einsum("ijil -> jl", ketandblocks)

def _node_state_phys_leg(node: Node) -> list[int]:
    """
    Finds the physical legs of a node of a state.

    Returns:
        list[int]: The indices of the physical legs of the node.
    """
    return node.open_legs

def _node_operator_input_leg(node: Node) -> list[int]:
    """
    Finds the legs of a node of the hamiltonian corresponding to the input.

    Returns:
        list[int]: The indices of the legs corresponding to input.
    """
    # Corr ket leg
    return list(range(node.nneighbours() + node.nopen_legs() // 2, node.nneighbours() + node.nopen_legs()))

def _node_operator_output_leg(node: Node) -> list[int]:
    """
    Finds the legs of a node of the hamiltonian corresponding to the output.
    
    Returns:
        list[int]: The indices of the legs corresponding to output.
    """
    # Corr bra leg
    return list(range(node.nneighbours(), node.nneighbours() + node.nopen_legs() // 2))

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
