"""
Modul that contains utility functions for contractions.

The functions here are used in mutliple different kinds of contractions.
"""

from __future__ import annotations
from typing import List, Union, Tuple, Callable

import numpy as np

from ..core.node import Node
from .tree_cach_dict import PartialTreeCachDict
from .local_contr import LocalContraction

def determine_index_with_ignored_leg(node: Node,
                                     neighbour_id: str,
                                     ignoring_node_id: str) -> int:
    """
    Determine the index of neighbour legs, while ignoring one leg.

    Sometimes when contracting all the neighbouring cached environments, we
    want to ignore the leg to a specific neighbour node. This means we do not
    want to contract that leg. This function determines the leg index of the
    current tensor, that should actually be contracted. This means earlier
    contractions are already taken into account.
    
    Args:
        node (Node): The node for which find the leg indices.
        neighbour_id (str): The identifier of the neighbour node.
        ignoring_node_id (str): The identifier of the neighbour leg to ignore.
    """
    neighbour_index = node.neighbour_index(neighbour_id)
    ignoring_index = node.neighbour_index(ignoring_node_id)
    assert ignoring_index != neighbour_index, "The next node should not be touched!"
    tensor_index_to_neighbour = int(ignoring_index < neighbour_index)
    return tensor_index_to_neighbour

def get_equivalent_legs(node1: Node,
                        node2: Node,
                        ignore_legs: Union[None,List[str],str] = None,
                        id_trafo: Union[None,Callable] = None,
                        ) -> Tuple[List[int],List[int]]:
    """
    Get the equivalent legs of two nodes. This is useful when contracting
     two nodes with equal neighbour identifiers, that may potentially be in
     different orders. Some neighbours may also be ignored.
    
    Args:
        node1 (Node): The first node.
        node2 (Node): The second node.
        ignore_legs (Union[None,List[str],str]): The legs to ignore as given
            by the ket identifiers.
        id_trafo (Union[None,Callable]): A function that transforms the
            node1's neighbour ids to the neighbour ids of node2. If None,
            the neighbour ids are assumed to be the same. Default is None.
    
    Returns:
        Tuple[List[int],List[int]]: The equivalent legs of the two nodes. This
            means the indeces of legs to the same neighbour are at the same
            position in each list.

    """
    if ignore_legs is None:
        ignore_legs = []
    elif isinstance(ignore_legs, str):
        ignore_legs = [ignore_legs]
    legs1 = []
    legs2 = []
    for neighbour_id in node1.neighbouring_nodes():
        if neighbour_id in ignore_legs:
            continue
        legs1.append(node1.neighbour_index(neighbour_id))
        if id_trafo is None:
            node2_neighbour_id = neighbour_id
        else:
            node2_neighbour_id = id_trafo(neighbour_id)
        legs2.append(node2.neighbour_index(node2_neighbour_id))
    return legs1, legs2

def contract_all_but_one_neighbour_block_to_ket(ket_tensor: np.ndarray,
                                                ket_node: Node,
                                                next_node_id: str,
                                                partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contract all neighbour blocks to the ket tensor, except for the one
    specified.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
    """
    nodes_tensors = [(ket_node, ket_tensor)]
    loc_contr = LocalContraction(nodes_tensors,
                                 partial_tree_cache,
                                 ignored_leg=next_node_id)
    return loc_contr()

def contract_all_neighbour_blocks_to_ket(ket_tensor: np.ndarray,
                                         ket_node: Node,
                                         partial_tree_cache: PartialTreeCachDict,
                                         order: Union[None,List[str]] = None
                                         ) -> np.ndarray:
    """
    Contract all neighbour blocks to the ket tensor.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        order (Union[None,List[str]]): The order in which the neighbour blocks
            should be contracted. If None, the order is simply pulled from the
            node's neighbouring nodes.

    Returns:
        np.ndarray: The resulting tensor::

             ______                 ______
            |      |____       ____|      |
            |      |n             n|      |
            |      | :           : |      |
            |      |____       ____|      |
            |  C1  |1      |      1|  C2  |
            |      |     __|__     |      |
            |      |____|     |____|      |
            |      |0   |  A  |   0|      |
            |______|    |_____|    |______|

        """
    if order is not None:
        num_neighbours = ket_node.nneighbours()
        if len(order) != num_neighbours:
            errstr = f"Order given does not match the number of neighbours {num_neighbours}!"
            raise ValueError(errstr)
    nodes_tensors = [(ket_node, ket_tensor)]
    loc_contr = LocalContraction(nodes_tensors,
                                 partial_tree_cache,
                                 neighbour_order=order)
    return loc_contr()   

def contract_all_but_one_neighbour_block_to_hamiltonian(hamiltonian_tensor: np.ndarray,
                                                        hamiltonian_node: Node,
                                                        next_node_id: str,
                                                        partial_tree_cache: PartialTreeCachDict,
                                                        operator_id_trafo: Callable | None
                                                        ) -> np.ndarray:
    """
    Contract all neighbour blocks to the Hamiltonian tensor.

    Args:
        hamiltonian_tensor (np.ndarray): The tensor of the Hamiltonian node.
        hamiltonian_node (Node): The Hamiltonian node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual leg points.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        operator_id_trafo (Callable | None): A function that transforms the
            neighbour identifiers used in the cache to the neighbour
            identifiers of the Hamiltonian node. If None, it is assumed to be
            the identity.

    Returns:
        np.ndarray: The resulting tensor::

                 _____       out      
                |     |____    
                |     |   2             
                |     |        |4       
                |     |     ___|__      
                |     |    |      |     
                |     |____|   H  |_____
                |     |    |      |     0
                |     |    |______|     
                |     |        |        
                |     |        |3    
                |     |                 
                |     |_____       
                |_____|    1      
                              in

    """
    nodes_tensors = [(hamiltonian_node, hamiltonian_tensor)]
    loc_contr = LocalContraction(nodes_tensors,
                                 partial_tree_cache,
                                 connection_index=1,
                                 ignored_leg=next_node_id,
                                 id_trafos=operator_id_trafo)
    return loc_contr()

def contract_all_neighbour_blocks_to_hamiltonian(hamiltonian_tensor: np.ndarray,
                                                 hamiltonian_node: Node,
                                                 partial_tree_cache: PartialTreeCachDict,
                                                 operator_id_trafo: Callable | None
                                                 ) -> np.ndarray:
    """
    Contract all neighbour blocks to the hamiltonian tensor.

    Args:
        hamiltonian_tensor (np.ndarray): The tensor of the hamiltonian node.
        hamiltonian_node (Node): The hamiltonian node.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        operator_id_trafo (Callable | None): A function that transforms the
            neighbour identifiers used in the cache to the neighbour
            identifiers of the Hamiltonian node. If None, it is assumed to be
            the identity.

    Returns:
        np.ndarray: The resulting tensor::

                 _____                   _____
                |     |____2      2_____|     |
                |     |                 |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|      |_____|     |
                |     | 1  |   H  |   1 |     |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |        |        |     |
                |     |                 |     |
                |     |_____       _____|     |
                |_____| 0           0   |_____|
                
                              
    """
    nodes_tensors = [(hamiltonian_node, hamiltonian_tensor)]
    loc_contr = LocalContraction(nodes_tensors,
                                 partial_tree_cache,
                                 connection_index=1,
                                 id_trafos=operator_id_trafo)
    return loc_contr()