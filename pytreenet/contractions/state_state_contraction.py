from __future__ import annotations

import numpy as np

from ..time_evolution.tdvp_util.partial_tree_cache import PartialTreeChachDict
from ..node import Node

def contract_two_ttn(ttn1: TreeTensorNetwork, ttn2: TreeTensorNetwork) -> complex:
    """
    Contracts two TreeTensorNetworks.

    Args:
        ttn1 (TreeTensorNetwork): The first TreeTensorNetwork.
        ttn2 (TreeTensorNetwork): The second TreeTensorNetwork.

    Returns:
        complex: The resulting scalar product <TTN1|TTN2>
    """
    return 

def contract_leafs(node_id: str, state1: TreeTensorNetworkState,
                   state2: TreeTensorNetworkState) -> np.ndarray:
    """
    Creates a SubTreeSandwichContraction for a leaf node.

    Args:
        node_id (str): The identifier of the leaf node.
        state1 (TreeTensorNetworkState): The first TTN state.
        state2 (TreeTensorNetworkState): The second TTN state.

    Returns:
        SubTreeSandwichContraction: The tensor resulting from the contraction:
                     _____
                ____|     |
                    |  B  |
                    |_____|
                       |
                       |
                     __|__
                ____|     |
                    |  A  |
                    |_____|
            
            where B is the tensor in state2 and A is the tensor in state1.
    """
    node1 = state1.nodes[node_id]
    node2 = state2.nodes[node_id]
    assert node1.is_leaf() and node2.is_leaf()
    errstr = "The leaf nodes must have exactly one open leg."
    assert len(node1.open_legs) == 1 and len(node2.open_legs) == 1, errstr
    tensor = np.tensordot(state1.tensors[node_id], state2.tensors[node_id],
                            axes=(node1.open_legs[0], node2.open_legs[0]))
    return tensor

def contract_subtrees_using_dictionary(node_id: str, next_node_id: str,
                                       state1: TreeTensorNetworkState,
                                       state2: TreeTensorNetworkState,
                                       dictionary: PartialTreeChachDict) -> np.ndarray:
    """
    The tensor with only two open legs, pointing to the same neighbour, after
     contracting a hole subtree. The subtrees attached to the other virtual
     legs are already contracted and stored in the dictionary.

    Args:
        node_id (str): The identifier of the node.
        next_node_id (str): The identifier of the node to which the remaining
         virtual legs point.
        state1 (TreeTensorNetworkState): The first TTN state.
        state2 (TreeTensorNetworkState): The second TTN state.
        dictionary (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
        
        Returns:
            np.ndarray: The resulting tensor. For example, if the nodes have
             two neighbours.

                         _____      ______
                    ____|     |____|      |
                        |  B  |    |      |
                        |_____|    |      |
                           |       |      |
                           |       |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|
    """
    ket_node = state1.nodes[node_id]
    ket_tensor = state1.tensors[node_id]
    ketblock_tensor = contract_all_neighbour_blocks_to_ket(ket_tensor, ket_node,
                                                           next_node_id, dictionary)
    bra_tensor = state2.tensors[node_id]
    bra_node = state2.nodes[node_id]
    return contract_bra_to_ket_and_blocks(bra_tensor, ketblock_tensor,
                                          bra_node, ket_node,
                                          next_node_id)

def contract_neighbour_block_to_ket(ket_tensor: np.ndarray,
                                    ket_node: Node,
                                    next_node_id: str,
                                    neighbour_id: str,
                                    partial_tree_cache: PartialTreeChachDict) -> np.ndarray:
    """
    Contracts the ket tensor, i.e. A in the diagrams, with one neighbouring
     block, C in the diagrams.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        neighbour_id (str): The identifier of the neighbour node which is the
         root node of the subtree that has already been contracted in the
         block C.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
    
    Returns:
        np.ndarray: The resulting tensor.
                                    ______
                               ____|      |
                                   |      |
                                   |      |
                                   |      |
                           |       |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|
    """
    cached_neighbour_tensor = partial_tree_cache.get_cached_tensor(neighbour_id,
                                                                   ket_node.identifier)
    next_node_index = ket_node.neighbour_index(next_node_id)
    neighbour_index = ket_node.neighbour_index(neighbour_id)
    assert next_node_index != neighbour_index, "The next node should not be touched!"
    tensor_index_to_neighbour = int(next_node_index > neighbour_index)
    return np.tensordot(ket_tensor, cached_neighbour_tensor,
                        axes=([tensor_index_to_neighbour],[0]))

def contract_all_neighbour_blocks_to_ket(ket_tensor: np.ndarray,
                                         ket_node: Node,
                                         next_node_id: str,
                                         partial_tree_cache: PartialTreeChachDict) -> np.ndarray:
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
        result_tensor = contract_neighbour_block_to_ket(result_tensor, ket_node,
                                                        next_node_id, neighbour_id,
                                                        partial_tree_cache)
    return result_tensor

def contract_bra_to_ket_and_blocks(bra_tensor: np.ndarray,
                                   ketblock_tensor: np.ndarray,
                                   bra_node: Node,
                                   ket_node: Node,
                                   next_node_id: str) -> np.ndarray:
    """
    Contracts the bra tensor, i.e. B in the diagrams, with the ketblock tensor,
     i.e. A and C in the diagrams.

    Args:
        bra_tensor (np.ndarray): The tensor of the bra node.
        ketblock_tensor (np.ndarray): The tensor resulting from the contraction
         of the ket node and its neighbouring blocks.
        bra_node (Node): The bra node.
        ket_node (Node): The ket node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.

    Returns:
        np.ndarray: The resulting tensor.

                         _____      ______
                    ____|     |____|      |
                        |  B  |    |      |
                        |_____|    |      |
                           |       |      |
                           |       |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|
    """
    legs_block = []
    legs_bra = []
    neighbours = bra_node.neighbouring_nodes()
    next_node_index = ket_node.neighbour_index(next_node_id)
    for neighbour_id in neighbours:
        if neighbour_id != next_node_id:
            ket_index = ket_node.neighbour_index(neighbour_id)
            legs_block.append(ket_index + 1 + int(next_node_index > ket_index))
            legs_bra.append(bra_node.neighbour_index(neighbour_id))
    legs_block.append(1)
    num_neighbours = bra_node.nneighbours()
    legs_bra.append(num_neighbours)
    return np.tensordot(ketblock_tensor, bra_tensor, axes=(legs_block, legs_bra))
