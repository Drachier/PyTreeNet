"""
This module provides functions to multiply a TTNS with a TTNO
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from copy import deepcopy

import numpy as np

from ..core.node import Node
from ..core.graph_node import find_children_permutation
from .tree_cach_dict import PartialTreeCachDict
from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                               contract_all_neighbour_blocks_to_ket,
                               get_equivalent_legs,
                               contract_all_but_one_neighbour_block_to_hamiltonian)
from ..util.tensor_splitting import tensor_qr_decomposition_pivot
from .state_operator_contraction import (contract_operator_tensor_ignoring_one_leg,
                                         _node_state_phys_leg,
                                         _node_operator_input_leg,
                                         contract_ket_tensor_ignoring_one_leg)
from ..util.tensor_splitting import SVDParameters

if TYPE_CHECKING:
    from ..ttns.ttns import TreeTensorNetworkState
    from ..ttno.ttno_class import TreeTensorNetworkOperator

__all__ = ['zipup']

def zipup(operator: TreeTensorNetworkOperator,
          state: TreeTensorNetworkState,
          svd_params: SVDParameters = None
          ) -> TreeTensorNetworkState:
    """
    Apply a TTNO to a TTNS.

    Args:
        operator (TreeTensorNetworkOperator): The TTNO to apply.
        state (TreeTensorNetworkState): The TTNS that TTNO acts on.
        svd_params (SVDParameters): The SVD parameters.

    Returns:    
        TreeTensorNetworkState: The result of the application of the TTNO to the TTNS.
    """
    if svd_params is None:
        svd_params = SVDParameters()
    dictionary = PartialTreeCachDict()
    # Getting a linear list of all identifiers
    computation_order = state.linearise()
    errstr = "The last element of the linearisation should be the root node."
    assert computation_order[-1] == state.root_id, errstr
    assert computation_order[-1] == operator.root_id, errstr
    resl_ttns = deepcopy(state)
    for node_id in computation_order[:-1]: # The last one is the root node
        node = resl_ttns.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        q,r = contract_any(node_id, parent_id,
                             resl_ttns, operator,
                             dictionary, svd_params)
        resl_ttns.nodes[node_id].link_tensor(q)
        resl_ttns.replace_tensor(node_id, q)
        dictionary.add_entry(node_id,parent_id,r)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    ttno_point = np.prod(operator.root[0].shape[:-2])/np.prod(operator.root[0].shape[-2:])
    ttns_point = np.prod(state.root[0].shape[:-1])/np.prod(state.root[0].shape[-1:])
    # Compare which contraction is more efficient, first TTNS then TTNO
    # (contract_node_with_environment) or first TTNO then TTNS
    # (contract_node_with_environment_2).
    if state.root[0].nneighbours() > 2 and ttno_point > ttns_point:
        tensor = contract_node_with_environment_2(resl_ttns.root_id,
                                            resl_ttns, operator,
                                           dictionary)
    else:
        tensor = contract_node_with_environment(resl_ttns.root_id,
                                            resl_ttns, operator,
                                           dictionary)
    root_node = resl_ttns.nodes[resl_ttns.root_id]
    root_node.link_tensor(tensor)
    resl_ttns.replace_tensor(resl_ttns.root_id, tensor)
    resl_ttns.canonical_form(resl_ttns.root_id)
    return resl_ttns

def contract_node_with_environment(node_id: str,
                                   state: TreeTensorNetworkState,
                                   operator: TreeTensorNetworkOperator,
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

                 _____                   _____
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
            ____|     |____|      |_____|     |____
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
    _, ham_legs = get_equivalent_legs(ket_node, op_node)
    ham_legs.append(_node_operator_input_leg(op_node)[0])
    block_legs = list(range(1,2*ket_node.nneighbours(),2))
    block_legs.append(0)

    kethamblock = np.tensordot(ket_neigh_block, op_tensor,
                               axes=(block_legs, ham_legs))
    return kethamblock

def contract_node_with_environment_2(node_id: str,
                                   state: TreeTensorNetworkState,
                                   operator: TreeTensorNetworkOperator,
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

                 _____                   _____
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
            ____|     |____|      |_____|     |____
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
    result_tensor = op_tensor
    for neighbour_id in op_node.neighbouring_nodes():
        cached_neighbour_tensor = dictionary.get_entry(neighbour_id, ket_node.identifier)
        result_tensor = np.tensordot(result_tensor, cached_neighbour_tensor,
                                     axes=([0], [1]))

    state_legs, _ = get_equivalent_legs(ket_node, op_node)
    perm = find_children_permutation(ket_node, op_node)
    state_legs = [state_legs[i] for i in perm]
    state_legs.append(-1)
    block_legs = list(range(2,2*ket_node.nneighbours()+2,2))
    block_legs.append(0)

    kethamblock1 = np.tensordot(result_tensor, ket_tensor,
                               axes=(block_legs, state_legs))
    kethamblock1 = np.moveaxis(kethamblock1, 0, -1)
    perm = find_children_permutation(op_node, ket_node)
    kethamblock1 = np.moveaxis(kethamblock1, perm, range(len(perm)))
    return kethamblock1

def contract_any(node_id: str, next_node_id: str,
                 state: TreeTensorNetworkState,
                 operator: TreeTensorNetworkOperator,
                 dictionary: PartialTreeCachDict,
                 svd_params: SVDParameters) -> np.ndarray:
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
                                                 dictionary, svd_params)

def contract_any_node_environment_but_one(ignored_node_id: str,
                                            ket_node: Node, ket_tensor: np.ndarray,
                                            op_node: Node, op_tensor: np.ndarray,
                                            dictionary: PartialTreeCachDict,
                                            svd_params: SVDParameters) -> np.ndarray:
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

    Returns:
        np.ndarray: The contracted tensor::

                                ______
                     __|__     |      |
              1 ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |_____
                       |       |      |
                     __|__     |      |
              0 ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|

    """
    if ket_node.is_leaf():
        new_state_tensor, cache_tensor = contract_leaf(ket_node, ket_tensor,
                             op_node, op_tensor, svd_params)
    else:
        new_state_tensor, cache_tensor = contract_subtrees_using_dictionary(ignored_node_id,
                                    ket_node, ket_tensor,
                                    op_node, op_tensor,
                                    dictionary, svd_params)
    
    return new_state_tensor, cache_tensor

def contract_leaf(state_node: Node,
                  state_tensor: np.ndarray,
                  operator_node: Node,
                  operator_tensor: np.ndarray,
                  svd_params: SVDParameters) -> np.ndarray:
    """
    Contracts for a leaf node the state, and operator.

    If the current subtree starts at a leaf, only the three tensors
    corresponding to that site must be contracted. Furthermore, the retained
    legs must point towards the leaf's parent.

    Args:
        state_node (Node): The node of the state.
        state_tensor (np.ndarray): The tensor of the state.
        operator_node (Node): The node of the operator.
        operator_tensor (np.ndarray): The tensor of the operator.

    Returns:
        np.ndarray: The contracted partial tree::
    
                       
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
    ham_ket = np.tensordot(state_tensor, operator_tensor,
                               axes=(_node_state_phys_leg(state_node),
                                     _node_operator_input_leg(operator_node)))
    q_legs = [2] 
    r_legs = [0,1] 
    q, r = tensor_qr_decomposition_pivot(ham_ket, q_legs, r_legs, svd_params)
    q = np.moveaxis(q, -1, 0)
    r = r.transpose([1,2,0])
    return q, r

def contract_subtrees_using_dictionary(ignored_node_id: str,
                                        ket_node: Node, ket_tensor: np.ndarray,
                                        op_node: Node, op_tensor: np.ndarray,
                                        dictionary: PartialTreeCachDict,
                                        svd_params: SVDParameters) -> np.ndarray:
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

    Returns:
        np.ndarray: The contracted tensor::

                       |        ______
                     __|__     |      |
              1 ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |_____
                       |       |      |
                     __|__     |      |
              0 ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|
    
    """
    ttno_point = np.prod(op_node.shape[:-2])/np.prod(op_node.shape[-2:])
    ttns_point = np.prod(ket_node.shape[:-1])/np.prod(ket_node.shape[-1:])
    if ket_node.nneighbours() > 2 and ttno_point > ttns_point:
        tensor = contract_all_but_one_neighbour_block_to_hamiltonian(op_tensor,
                                                                     op_node,
                                                                     ignored_node_id,
                                                                     dictionary)
        tensor = contract_ket_tensor_ignoring_one_leg(tensor,
                                                      op_node,
                                                      ket_tensor,
                                                      ket_node,
                                                      ignored_node_id)
    else:
        tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                            ket_node,
                                                            ignored_node_id,
                                                            dictionary)
        tensor = contract_operator_tensor_ignoring_one_leg(tensor,
                                                        ket_node,
                                                        op_tensor,
                                                        op_node,
                                                        ignored_node_id)
    q_legs = list(range(1,ket_node.nneighbours()))
    q_legs.append(ket_node.nneighbours()+1)
    r_legs= [0,ket_node.nneighbours()]
    q, r = tensor_qr_decomposition_pivot(tensor, q_legs, r_legs, svd_params)
    q = np.moveaxis(q, -1, 0)
    r = r.transpose([1,2,0])
    return q, r
