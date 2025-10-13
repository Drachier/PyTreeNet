"""
This module contains functions to find effective Hamiltonians.

The functions use cached tensors to be more efficient.
"""

from typing import Tuple

from numpy import ndarray, tensordot, transpose

from ..core.node import Node
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..util.tensor_util import tensor_matricisation_half
from .local_contr import LocalContraction, FinalTransposition

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
    nodes_tensors = [(hamiltonian_node, hamiltonian_tensor)]
    loc_contr = LocalContraction(nodes_tensors,
                                 tensor_cache,
                                 node_identifier=state_node.identifier,
                                 neighbour_order=state_node.neighbouring_nodes(),
                                 connection_index=1)
    return loc_contr(transpose_option=FinalTransposition.HAMILTONIAN)

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

def contract_all_except_two_nodes(state_target: Node,
                                  target_node: Node,
                                  target_tensor: ndarray,
                                  state_next: Node,
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
        state_target (Node): The node of the state tensor of the target site.
        target_node (Node): The target node.
        target_tensor (ndarray): The tensor of the target node.
        state_next (Node): The node of the state tensor of the next site.
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
    loc_contr_traget = LocalContraction([(target_node, target_tensor)],
                                        tensor_cache,
                                        ignored_leg=next_node_id,
                                        node_identifier=state_target.identifier,
                                        neighbour_order=state_target.neighbouring_nodes(),
                                        connection_index=1)
    loc_contr_next = LocalContraction([(next_node, next_tensor)],
                                        tensor_cache,
                                        ignored_leg=target_node_id,
                                        node_identifier=state_next.identifier,
                                        neighbour_order=state_next.neighbouring_nodes(),
                                        connection_index=1)
    h_eff = loc_contr_traget.contract_to_other(loc_contr_next,
                                               FinalTransposition.HAMILTONIAN)
    return h_eff

def get_effective_two_site_hamiltonian_nodes(state_target: Node,
                                             target_node: Node,
                                             target_tensor: ndarray,
                                             state_next: Node,
                                             next_node: Node,
                                             next_tensor: ndarray,
                                             tensor_cache: PartialTreeCachDict) -> ndarray:
    """
    Obtains the effective two-site Hamiltonian as a matrix.

    Args:
        state_target (Node): The node of the state tensor of the target site.
        target_node (Node): The node of the target Hamiltonian tensor.
        target_tensor (ndarray): The tensor of the target node.
        state_next (Node): The node of the state tensor of the next site.
        next_node (Node): The node of the next Hamiltonian tensor.
        next_tensor (ndarray): The tensor of the next node.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective two-site Hamiltonian

    """
    tensor = contract_all_except_two_nodes(state_target,
                                          target_node,
                                          target_tensor,
                                          state_next,
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
    state_target = state.nodes[target_node_id]
    state_next = state.nodes[next_node_id]
    return get_effective_two_site_hamiltonian_nodes(state_target,
                                                    target_node,
                                                    target_tensor,
                                                    state_next,
                                                    next_node,
                                                    next_tensor,
                                                    tensor_cache)
