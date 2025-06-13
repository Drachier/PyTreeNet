"""
This module contains functions to find effective Hamiltonians.

The functions use cached tensors to be more efficient.
"""

from typing import Tuple

from numpy import ndarray, tensordot, transpose

from ..core.graph_node import GraphNode
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..util.tensor_util import tensor_matricisation_half

def find_tensor_leg_permutation(state_node: GraphNode,
                                hamiltonian_node: GraphNode
                                ) -> Tuple[int,...]:
    """
    Find the correct permutation to permute the effective hamiltonian
    tensor to fit with the state tensor legs.
    
    After contracting all the cached tensors to the site Hamiltonian, the
    legs of the resulting tensor are in the order of the Hamiltonian TTNO.
    However, they need to be permuted to match the legs of the site's
    state tensor. Such that the two can be easily contracted.

    Args:
        state_node (GraphNode): The node of the state tensor.
        hamiltonian_node (GraphNode): The node of the Hamiltonian tensor.
    
    Returns:
        Tuple[int,...]: The permutation to apply to the effective
            Hamiltonian tensor to match the legs of the state tensor.

    """
    permutation = [hamiltonian_node.neighbour_index(neighbour_id)
                   for neighbour_id in state_node.neighbouring_nodes()]
    output_legs = []
    input_legs = []
    for hamiltonian_index in permutation:
        output_legs.append(2*hamiltonian_index+3)
        input_legs.append(2*hamiltonian_index+2)
    output_legs.append(0)
    input_legs.append(1)
    output_legs.extend(input_legs)
    return tuple(output_legs)

def contract_all_except_node(state_node: GraphNode,
                             hamiltonian_node: GraphNode,
                             hamiltonian_tensor: ndarray,
                             tensor_cache: PartialTreeCachDict) -> ndarray:
    """
    Contract bra, ket and hamiltonian for all but one node into that
    node's Hamiltonian tensor.

    Uses the cached trees to contract the bra, ket, and hamiltonian
    tensors for all nodes in the trees apart from the given target node.
    All the resulting tensors are contracted to the hamiltonian tensor
    corresponding to the target node.

    Args:
        state_node (GraphNode): The node of the state tensor.
        hamiltonian_node (GraphNode): The node of the Hamiltonian tensor.
        hamiltonian_tensor (ndarray): The Hamiltonian tensor that will be
            contracted with the cached tensors.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.
    
    Returns:
        ndarray: The tensor resulting from the contraction::
        
             _____        out        _____
            |     |____n-1    0_____|     |
            |     |                 |     |
            |     |        |n       |     |
            |     |     ___|__      |     |
            |     |    |      |     |     |
            |     |____|      |_____|     |
            |     |    |   H  |     |     |
            |     |    |______|     |     |
            |     |        |        |     |
            |     |        |2n+1    |     |
            |     |                 |     |
            |     |_____       _____|     |
            |_____|  2n         n+1 |_____|
                            in

            where n is the number of neighbours of the node.

    """
    target_node_id = state_node.identifier
    neighbours = hamiltonian_node.neighbouring_nodes()
    for neighbour_id in neighbours:
        cached_tensor = tensor_cache.get_entry(neighbour_id,
                                                target_node_id)
        hamiltonian_tensor = tensordot(hamiltonian_tensor,
                                       cached_tensor,
                                       axes=((0,1)))
    # Transposing to have correct leg order
    axes = find_tensor_leg_permutation(state_node, hamiltonian_node)
    output_tensor = transpose(hamiltonian_tensor, axes=axes)
    return output_tensor

def get_effective_single_site_hamiltonian_nodes(state_node: GraphNode,
                                                hamiltonian_node: GraphNode,
                                                hamiltonian_tensor: ndarray,
                                                tensor_cache: PartialTreeCachDict) -> ndarray:
    """
    Obtains the effective site Hamiltonian as a matrix.

    Args:
        state_node (GraphNode): The node of the state tensor.
        hamiltonian_node (GraphNode): The node of the Hamiltonian tensor.
        hamiltonian_tensor (ndarray): The Hamiltonian tensor that will be
            contracted with the cached tensors.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective site Hamiltonian

    """
    tensor = contract_all_except_node(state_node,
                                      hamiltonian_node,
                                      hamiltonian_tensor,
                                      tensor_cache)
    return tensor_matricisation_half(tensor)

def get_effective_single_site_hamiltonian(node_id: str,
                                          state: TreeTensorNetworkState,
                                          hamiltonian: TreeTensorNetworkOperator,
                                          tensor_cache: PartialTreeCachDict) -> ndarray:
    """
    Obtains the effective site Hamiltonian as a matrix.

    Args:
        node_id (str): The identifier of the node for which the effective
            Hamiltonian is to be found.
        state (TreeTensorNetworkState): The state of the system.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the
            system.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective site Hamiltonian

    """
    state_node = state.nodes[node_id]
    hamiltonian_node, hamiltonian_tensor = hamiltonian[node_id]
    return get_effective_single_site_hamiltonian_nodes(state_node,
                                                       hamiltonian_node,
                                                       hamiltonian_tensor,
                                                       tensor_cache)

def get_effective_bond_hamiltonian_tensor(
                                         state_node: GraphNode,
                                         tensor_cache: PartialTreeCachDict
                                         ) -> ndarray:
    """
    Obtains the effective bond Hamiltonian as a matrix.

    Args:
        state_node (GraphNode): The node of the state tensor, representing the
            bond for which no equivalent node exists in the TTNO.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective bond Hamiltonian

             _____        out        _____
            |     |____0      1_____|     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |_________________|     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |_____       _____|     |
            |_____|  2         3    |_____|
                            in    
    
    """
    neighbours = state_node.neighbouring_nodes()
    if len(neighbours) != 2:
        raise ValueError("The effective bond Hamiltonian can only be "
                         "computed for a bond with two neighbours.")
    nghbrp = state_node.parent
    nghbrc = state_node.children[0]
    tensorp = tensor_cache.get_entry(nghbrp,
                                     nghbrc)
    tensorc = tensor_cache.get_entry(nghbrc,
                                     nghbrp)
    # Contract the Hamiltonian legs
    tensor = tensordot(tensorp,
                       tensorc,
                       axes=(1, 1))
    tensor = transpose(tensor, axes=[1, 3, 0, 2])
    return tensor

def get_effective_bond_hamiltonian_nodes(
                                         state_node: GraphNode,
                                         tensor_cache: PartialTreeCachDict
                                         ) -> ndarray:
    """
    Obtains the effective bond Hamiltonian as a matrix.

    Args:
        state_node (GraphNode): The node of the state tensor, representing the
            bond for which no equivalent node exists in the TTNO.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective bond Hamiltonian as a matrix.
    
    """
    tensor = get_effective_bond_hamiltonian_tensor(state_node,
                                                    tensor_cache)
    return tensor_matricisation_half(tensor)

def get_effective_bond_hamiltonian(bond_node_id: str,
                                   state: TreeTensorNetworkState,
                                   tensor_cache: PartialTreeCachDict
                                   ) -> ndarray:
    """
    Obtains the effective bond Hamiltonian as a matrix.

    Args:
        bond_node_id (str): The identifier of the node representing the bond
            for which the effective Hamiltonian is to be found.
        state (TreeTensorNetworkState): The state of the system.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective bond Hamiltonian as a matrix.
    """
    state_node = state.nodes[bond_node_id]
    return get_effective_bond_hamiltonian_nodes(state_node, tensor_cache)
