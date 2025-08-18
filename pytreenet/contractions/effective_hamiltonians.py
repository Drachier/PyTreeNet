"""
This module contains functions to find effective Hamiltonians.

The functions use cached tensors to be more efficient.
"""

from typing import Tuple

from numpy import ndarray, tensordot, transpose
import numpy as np

from ..core.node import Node
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..util.tensor_util import tensor_matricisation_half
from .contraction_util import contract_all_but_one_neighbour_block_to_hamiltonian
from ..util.ttn_exceptions import NotCompatibleException

def find_tensor_leg_permutation(state_node: Node,
                                hamiltonian_node: Node
                                ) -> Tuple[int,...]:
    """
    Find the correct permutation to permute the effective hamiltonian
    tensor to fit with the state tensor legs.
    
    After contracting all the cached tensors to the site Hamiltonian, the
    legs of the resulting tensor are in the order of the Hamiltonian TTNO.
    However, they need to be permuted to match the legs of the site's
    state tensor. Such that the two can be easily contracted.

    Args:
        state_node (Node): The node of the state tensor.
        hamiltonian_node (Node): The node of the Hamiltonian tensor.
    
    Returns:
        Tuple[int,...]: The permutation to apply to the effective
            Hamiltonian tensor to match the legs of the state tensor.

    """
    permutation = [hamiltonian_node.neighbour_index(neighbour_id)
                   for neighbour_id in state_node.neighbouring_nodes()]
    n_open = state_node.nopen_legs()
    output_legs = []
    input_legs = []
    for hamiltonian_index in permutation:
        output_legs.append(2*n_open + 2*hamiltonian_index + 1)
        input_legs.append(2*n_open + 2*hamiltonian_index)
    output_legs.extend(list(range(n_open)))
    input_legs.extend(list(range(n_open, 2*n_open)))
    output_legs.extend(input_legs)
    return tuple(output_legs)

def contract_all_except_node(state_node: Node,
                             hamiltonian_node: Node,
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

def get_effective_single_site_hamiltonian_nodes(state_node: Node,
                                                hamiltonian_node: Node,
                                                hamiltonian_tensor: ndarray,
                                                tensor_cache: PartialTreeCachDict) -> ndarray:
    """
    Obtains the effective site Hamiltonian as a matrix.

    Args:
        state_node (Node): The node of the state tensor.
        hamiltonian_node (Node): The node of the Hamiltonian tensor.
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
                                         state_node: Node,
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
                                         state_node: Node,
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

def contract_all_except_two_nodes(state_node: Node,
                                  target_node: Node,
                                  target_tensor: ndarray,
                                  next_node: Node,
                                  next_tensor: ndarray,
                                  tensor_cache: PartialTreeCachDict
                                  ) -> ndarray:
    """
    Contracts the nodes for all but two sites.

    Uses all cached tensors to contract bra, ket, and hamiltonian tensors
    of all nodes except for the two given nodes. Of these nodes only the
    Hamiltonian nodes are contracted.
    IMPORTANT: This function assumes that the given nodes are already
    contracted in the TTNS.

    Args:
        state_node (Node): The node of the state tensor.
        target_node (Node): The target node.
        target_tensor (ndarray): The tensor of the target node.
        next_node (Node): The next node.
        next_tensor (ndarray): The tensor of the next node.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        np.ndarray: The resulting effective two-site Hamiltonian tensor::

             _____                out              _____
            |     |____n-1                  0_____|     |
            |     |                               |     |
            |     |        |n           |         |     |
            |     |     ___|__        __|___      |     |
            |     |    |      |      |      |     |     |
            |     |____|      |______|      |_____|     |
            |     |    |  H_1 |      | H_2  |     |     |
            |     |    |______|      |______|     |     |
            |     |        |            |         |     |
            |     |        |2n+1        |         |     |
            |     |                               |     |
            |     |_____                     _____|     |
            |_____|  2n                       n+1 |_____|
                                    in
    
    """
    # Contract all but one neighbouring block of each node
    target_node_id = target_node.identifier
    next_node_id = next_node.identifier
    target_block = contract_all_but_one_neighbour_block_to_hamiltonian(target_tensor,
                                                                        target_node,
                                                                        next_node_id,
                                                                        tensor_cache)

    next_block = contract_all_but_one_neighbour_block_to_hamiltonian(next_tensor,
                                                                        next_node,
                                                                        target_node_id,
                                                                        tensor_cache)
    # Contract the two blocks
    h_eff = np.tensordot(target_block, next_block, axes=(0,0))
    # Now we need to sort the legs to fit with the underlying TTNS.
    # Note that we assume, the two tensors have already been contracted.
    leg_permutation = _determine_two_site_leg_permutation(state_node, target_node,
                                                                next_node)
    return h_eff.transpose(leg_permutation)

def create_two_site_id(node_id: str, next_node_id: str) -> str:
    """
    Create the identifier of a two site node obtained from contracting
    the two note with the input identifiers.
    """
    return "TwoSite_" + node_id + "_contr_" + next_node_id

def _determine_two_site_leg_permutation(state_node: Node,
                                        target_node: Node,
                                        next_node: Node) -> Tuple[int]:
    """
    Determine the permutation of the effective Hamiltonian tensor.
    
    This is the leg permutation required on the two-site effective
    Hamiltonian tensor to fit with the underlying TTNS, assuming
    the two sites have already been contracted in the TTNS.

    Args:
        state (TreeTensorNetworkState): The state of the system.
        target_node (Node): The target node.
        next_node (Node): The next node.
    
    Returns:
        Tuple[int]: The permutation of the legs of the two-site effective
            Hamiltonian tensor.
    """
    neighbours_target = target_node.neighbouring_nodes()
    neighbours_next = next_node.neighbouring_nodes()
    neighbours_two_site = state_node.neighbouring_nodes()
    next_node_id = next_node.identifier
    
    # Determine the permutation of the legs
    input_legs = []
    for neighbour_id in neighbours_two_site:
        if neighbour_id in neighbours_target:
            block_leg = _find_block_leg_target_node(target_node,
                                                            next_node_id,
                                                            neighbour_id)
        elif neighbour_id in neighbours_next:
            block_leg = _find_block_leg_next_node(target_node,
                                                        next_node,
                                                        neighbour_id)
        else:
            errstr = "The two-site Hamiltonian has a neighbour that is not a neighbour of the two sites."
            raise NotCompatibleException(errstr)
        input_legs.append(block_leg)
    output_legs = [leg + 1 for leg in input_legs]
    target_num_neighbours = target_node.nneighbours()
    output_legs = output_legs + [0,2*target_num_neighbours] # physical legs
    input_legs = input_legs + [1,2*target_num_neighbours + 1] # physical legs
    # As in matrices, the output legs are first
    return tuple(output_legs + input_legs)

def _find_block_leg_target_node(target_node: Node,
                                next_node_id: str,
                                neighbour_id: str) -> int:
    """
    Determines the leg index of the input leg of the contracted subtree
    block on the effective hamiltonian tensor corresponding to a given
    neighbour of the target node.

    Args:
        target_node (Node): The target node.
        next_node_id (str): The id of the next node.
        neighbour_id (str): The id of the neighbour of the target node.
    
    Returns:
        int: The leg index of the input leg of the contracted subtree
            block on the effective hamiltonian tensor.
    """
    index_next_node = target_node.neighbour_index(next_node_id)
    ham_neighbour_index = target_node.neighbour_index(neighbour_id)
    constant = int(ham_neighbour_index < index_next_node)
    return 2 * (ham_neighbour_index + constant)

def _find_block_leg_next_node(target_node: Node,
                              next_node: Node,
                              neighbour_id: str) -> int:
    """
    Determines the leg index of the input leg of the contracted subtree
    bnlock on the effective hamiltonian tensor corresponding to a given
    eighbour of the next node.

    Args:
        target_node (Node): The target node.
        next_node (Node): The next node.
        neighbour_id (str): The id of the neighbour of the next node.
    
    Returns:
        int: The leg index of the input leg of the contracted subtree
            block on the effective hamiltonian tensor.
    """
    # Luckily the situation is pretty much the same so we can reuse most
    # of the code.
    target_node_id = target_node.identifier
    leg_index_temp = _find_block_leg_target_node(next_node,
                                                        target_node_id,
                                                        neighbour_id)
    target_node_numn = target_node.nneighbours()
    return 2 * target_node_numn + leg_index_temp

def get_effective_two_site_hamiltonian_nodes(state_node: Node,
                                             target_node: Node,
                                             target_tensor: ndarray,
                                             next_node: Node,
                                             next_tensor: ndarray,
                                             tensor_cache: PartialTreeCachDict) -> ndarray:
    """
    Obtains the effective two-site Hamiltonian as a matrix.

    Args:
        state_node (Node): The node of the state tensor.
        target_node (Node): The node of the target Hamiltonian tensor.
        target_tensor (ndarray): The tensor of the target node.
        next_node (Node): The node of the next Hamiltonian tensor.
        next_tensor (ndarray): The tensor of the next node.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective two-site Hamiltonian

    """
    tensor = contract_all_except_two_nodes(state_node,
                                          target_node,
                                          target_tensor,
                                          next_node,
                                          next_tensor,
                                          tensor_cache)
    return tensor_matricisation_half(tensor)

def get_effective_two_site_hamiltonian(target_node_id: str,
                                       next_node_id: str,
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
    target_node, target_tensor = hamiltonian[target_node_id]
    next_node, next_tensor = hamiltonian[next_node_id]
    two_site_id = create_two_site_id(target_node_id, next_node_id)
    state_node = state.nodes[two_site_id]
    return get_effective_two_site_hamiltonian_nodes(state_node,
                                                    target_node,
                                                    target_tensor,
                                                    next_node,
                                                    next_tensor,
                                                    tensor_cache)
