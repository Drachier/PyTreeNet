"""
This module contains functions to find effective Hamiltonians.

The functions use cached tensors to be more efficient.
"""

from typing import Tuple

from numpy import ndarray, tensordot, transpose

from ..core.graph_node import GraphNode
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..contractions.sandwich_caching import SandwichCache
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

def find_tensor_leg_permutation_from_id(node_id: str,
                                        state: TreeTensorNetworkState,
                                        hamiltonian: TreeTensorNetworkOperator
                                        ) -> Tuple[int,...]:
    """
    Find the correct permutation to permute the effective hamiltonian
    tensor to fit with the state tensor legs.
    
    After contracting all the cached tensors to the site Hamiltonian, the
    legs of the resulting tensor are in the order of the Hamiltonian TTNO.
    However, they need to be permuted to match the legs of the site's
    state tensor. Such that the two can be easily contracted.

    Args:
        node_id (str): The identifier of the node.
        state (TreeTensorNetworkState): The state tensor network.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian tensor
            network operator.
    
    Returns:
        Tuple[int,...]: The permutation to apply to the effective
            Hamiltonian tensor to match the legs of the state tensor.

    """
    state_node = state.nodes[node_id]
    hamiltonian_node = hamiltonian.nodes[node_id]
    return find_tensor_leg_permutation(state_node, hamiltonian_node)

def contract_all_except_node(target_node_id: str,
                             state: TreeTensorNetworkState,
                             hamiltonian: TreeTensorNetworkOperator,
                             tensor_cache: SandwichCache) -> ndarray:
    """
    Contract bra, ket and hamiltonian for all but one node into that
    node's Hamiltonian tensor.

    Uses the cached trees to contract the bra, ket, and hamiltonian
    tensors for all nodes in the trees apart from the given target node.
    All the resulting tensors are contracted to the hamiltonian tensor
    corresponding to the target node.

    Args:
        target_node_id (str): The node which is not to be part of the
            contraction.
        state (TreeTensorNetworkState): The state of the system.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the
            system.
        tensor_cache (SandwichCache): The cache of environment tensors.
    
    Returns:
        ndarray: The tensor resulting from the contraction::
        
                _____       out         _____
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
    target_node = hamiltonian.nodes[target_node_id]
    neighbours = target_node.neighbouring_nodes()
    tensor = hamiltonian.tensors[target_node_id]
    for neighbour_id in neighbours:
        cached_tensor = tensor_cache.get_entry(neighbour_id,
                                                            target_node_id)
        tensor = tensordot(tensor, cached_tensor,
                                axes=((0,1)))
    # Transposing to have correct leg order
    axes = find_tensor_leg_permutation_from_id(target_node_id,
                                               state,
                                               hamiltonian)
    tensor = transpose(tensor, axes=axes)
    return tensor

def get_effective_single_site_hamiltonian(node_id: str,
                                          state: TreeTensorNetworkState,
                                          hamiltonian: TreeTensorNetworkOperator,
                                          tensor_cache: SandwichCache) -> ndarray:
    """
    Obtains the effective site Hamiltonian as a matrix.

    Args:
        node_id (str): The identifier of the node for which the effective
            Hamiltonian is to be found.
        state (TreeTensorNetworkState): The state of the system.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the
            system.
        tensor_cache (SandwichCache): The cache of environment tensors.

    Returns:
        ndarray: The effective site Hamiltonian

    """
    tensor = contract_all_except_node(node_id,
                                      state,
                                      hamiltonian,
                                      tensor_cache)
    return tensor_matricisation_half(tensor)
