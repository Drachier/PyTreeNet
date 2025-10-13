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
from typing import Union, Callable, TYPE_CHECKING

import numpy as np

from .tree_cach_dict import PartialTreeCachDict
from .local_contr import LocalContraction

if TYPE_CHECKING:
    from ..core.node import Node
    from ..ttns.ttns import TreeTensorNetworkState

__all__ = ['contract_two_ttns', 'scalar_product']
        
def contract_two_ttns(ttn1: TreeTensorNetworkState,
                      ttn2: TreeTensorNetworkState,
                      id_trafo: Union[None,Callable] = None
                      ) -> complex:
    """
    Contracts two TreeTensorNetworks.

    Args:
        ttn1 (TreeTensorNetwork): The first TreeTensorNetwork.
        ttn2 (TreeTensorNetwork): The second TreeTensorNetwork.
        id_trafo (Union[None,Callable]): A function to transform the node
            identifiers of ttn1 into the node identifiers of ttn2. If None,
            it is assumed that the identifiers are the same.
            Defaults to None.

    Returns:
        complex: The resulting scalar product <TTN1|TTN2>
    """
    dictionary = PartialTreeCachDict()
    computation_order = ttn1.linearise() # Getting a linear list of all identifiers
    id_trafo = id_trafo if id_trafo is not None else lambda x: x
    for node_id in computation_order[:-1]: # The last one is the root node
        node = ttn1.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any(node_id, parent_id,
                             ttn1, ttn2,
                             dictionary,
                             id_trafo=id_trafo)
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    return contract_node_with_environment(ttn1.root_id,
                                            ttn1, ttn2,
                                            dictionary,
                                            id_trafo=id_trafo)

def scalar_product(ttn: TreeTensorNetworkState) -> complex:
    """
    Computes the scalar product of a TreeTensorNetworkState with itself.

    Args:
        ttn (TreeTensorNetwork): The TreeTensorNetwork.

    Returns:
        complex: The resulting scalar product <TTN|TTN>
    """
    dictionary = PartialTreeCachDict()
    computation_order = ttn.linearise() # Getting a linear list of all identifiers
    for node_id in computation_order[:-1]: # The last one is the root node
        node, tensor = ttn[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any_nodes(parent_id,
                                   node, node,
                                   tensor, tensor.conj(),
                                   dictionary)
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    root_node, root_tensor = ttn[ttn.root_id]
    return contract_node_with_environment_nodes(root_node, root_tensor,
                                                root_node, root_tensor.conj(),
                                                dictionary)

def contract_node_with_environment(node_id: str,
                                   state1: TreeTensorNetworkState,
                                   state2: TreeTensorNetworkState,
                                   dictionary: PartialTreeCachDict,
                                   id_trafo: Union[None,Callable] = None
                                   ) -> np.ndarray:
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
        id_trafo (Union[None,Callable]): A function to transform the node
            identifiers of state1 into the node identifiers of state2. If
            None, it is assumed that the identifiers are the same.
            Defaults to None.
    
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
    if id_trafo is None:
        bra_node, bra_tensor = state2[node_id]
    else:
        bra_node, bra_tensor = state2[id_trafo(node_id)]
    return contract_node_with_environment_nodes(ket_node, ket_tensor,
                                                bra_node, bra_tensor,
                                                dictionary,
                                                id_trafo=id_trafo)

def contract_node_with_environment_nodes(ket_node: Node,
                                         ket_tensor: np.ndarray,
                                         bra_node: Node,
                                         bra_tensor: np.ndarray,
                                         dictionary: PartialTreeCachDict,
                                         id_trafo: Union[None,Callable] = None
                                         ) -> complex:
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
        id_trafo (Union[None,Callable]): A function to transform the node
            identifiers of the ket node into the node identifiers of the bra
            node. If None, it is assumed that the identifiers are the same.
            Defaults to None.
    
    Returns:
        complex: The resulting tensor. A and B are the tensors in state1 and
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
    nodes_tensors = [(ket_node, ket_tensor),
                     (bra_node, bra_tensor)]
    local_contr = LocalContraction(nodes_tensors,
                                   dictionary,
                                   id_trafos=[None,id_trafo])
    return local_contr.contract_to_scalar()

def contract_any(node_id: str, next_node_id: str,
                 state1: TreeTensorNetworkState,
                 state2: TreeTensorNetworkState,
                 dictionary: PartialTreeCachDict,
                 state2_conj: bool = False,
                 id_trafo: Union[Callable,None] = None
                 ) -> np.ndarray:
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
        state2_conj (bool): Whether to use the complex conjugate of the tensors
            in state2. Defaults to False.
        id_trafo (Union[Callable,None], optional): A function to transform the
            node identifiers of state1 into the node identifiers of stat2. If
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
    node1, tensor1 = state1[node_id]
    if id_trafo is None:
        node2, tensor2 = state2[node_id]
    else:
        node2, tensor2 = state2[id_trafo(node_id)
                                if id_trafo is not None else node_id]
    if state2_conj:
        tensor2 = np.conjugate(tensor2)
    return contract_any_nodes(next_node_id, node1, node2,
                              tensor1, tensor2, dictionary,
                              id_trafo=id_trafo)

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
    nodes_tensors = [(node1, tensor1),
                     (node2, tensor2)]
    local_contr = LocalContraction(nodes_tensors,
                                   dictionary,
                                   id_trafos=[None,id_trafo],
                                   ignored_leg=next_node_id)
    return local_contr()
