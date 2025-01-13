"""
Modul that contains utility functions for contractions.

The functions here are used in mutliple different kinds of contractions.
"""
# TODO: Refactorise to avoid code duplication for ket and hamiltonian contractions.

from __future__ import annotations
from typing import List, Union, Tuple, Callable

import numpy as np

from ..core.node import Node
from .tree_cach_dict import PartialTreeCachDict

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

def contract_neighbour_block_to_ket(ket_tensor: np.ndarray,
                                    ket_node: Node,
                                    neighbour_id: str,
                                    partial_tree_cache: PartialTreeCachDict,
                                    tensor_leg_to_neighbour: Union[None,int]=None) -> np.ndarray:
    """
    Contracts the ket tensor with one neighbouring block.

    This means A in the diagram is contracted with C in the diagram.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        neighbour_id (str): The identifier of the neighbour node which is the
            root node of the subtree that has already been contracted and is
            saved in the dictionary.
        tensor_leg_to_neighbour (int): The index of the leg of the ket tensor
            that points to the neighbour block and is thus to be contracted.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing
            the already contracted subtrees. The tensors in here can have an
            arbitrary number of legs, but the first leg is the one that is
            contracted with the ket tensor.
    
    Returns:
        np.ndarray: The resulting tensor::

                                    ______
                               ____|      |
                               . n |      |
                               :   |      |
                               ____|      |
                           |     1 |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |  0 |      |
                        |_____|    |______|

    """
    cached_neighbour_tensor = partial_tree_cache.get_entry(neighbour_id,
                                                           ket_node.identifier)
    if tensor_leg_to_neighbour is None:
        tensor_leg_to_neighbour = ket_node.neighbour_index(neighbour_id)
    return np.tensordot(ket_tensor, cached_neighbour_tensor,
                        axes=([tensor_leg_to_neighbour],[0]))

def contract_neighbour_block_to_ket_ignore_one_leg(ket_tensor: np.ndarray,
                                                   ket_node: Node,
                                                   neighbour_id: str,
                                                   ignoring_node_id: str,
                                                   partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts the ket tensor with one neighbouring block, ignoring one leg.

    This means the ket tensor, i.e. A in the diagrams, is contracted with one
    neighbouring block, C in the diagrams, ignoring one leg.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        neighbour_id (str): The identifier of the neighbour node which is the
            root node of the subtree that has already been contracted and is
            saved in the dictionary.
        ignoring_node_id (str): The identifier of the node to which the virtual
            leg should not point.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing
            the already contracted subtrees. The tensors in here can have an
            arbitrary number of legs, but the first leg is the one that is
            contracted with the ket tensor.

    Returns:
        np.ndarray: The resulting tensor::

                                    ______
                               ____|      |
                               . n |      |
                               :   |      |
                               ____|      |
                           |     1 |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |  0 |      |
                        |_____|    |______|

    """
    tensor_index_to_neighbour = determine_index_with_ignored_leg(ket_node,
                                                                 neighbour_id,
                                                                 ignoring_node_id)
    return contract_neighbour_block_to_ket(ket_tensor, ket_node,
                                           neighbour_id,
                                           partial_tree_cache,
                                           tensor_leg_to_neighbour=tensor_index_to_neighbour)        

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
    result_tensor = ket_tensor
    for neighbour_id in ket_node.neighbouring_nodes():
        if neighbour_id != next_node_id:
            result_tensor = contract_neighbour_block_to_ket_ignore_one_leg(result_tensor,
                                                                           ket_node,
                                                                           neighbour_id,
                                                                           next_node_id,
                                                                           partial_tree_cache)
    return result_tensor

def contract_all_neighbour_blocks_to_ket(ket_tensor: np.ndarray,
                                         ket_node: Node,
                                         partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contract all neighbour blocks to the ket tensor.

    Args:
        ket_tensor (np.ndarray): The tensor of the ket node.
        ket_node (Node): The ket node.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.

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
    result_tensor = ket_tensor
    for neighbour_id in ket_node.neighbouring_nodes():
        # A the neighbours are the same as the leg order, the tensor_leg_to_neighbour
        # is always 0.
        result_tensor = contract_neighbour_block_to_ket(result_tensor,
                                                        ket_node,
                                                        neighbour_id,
                                                        partial_tree_cache,
                                                        tensor_leg_to_neighbour=0)
    return result_tensor

def contract_neighbour_block_to_hamiltonian(hamiltonian_tensor: np.ndarray,
                                            hamiltonian_node: Node,
                                            neighbour_id: str,
                                            partial_tree_cache: PartialTreeCachDict,
                                            tensor_leg_to_neighbour: Union[None,int]=None) -> np.ndarray:
    """
    Contract the Hamiltonian tensor, i.e. H in the diagrams, with one neighbouring
     block, C in the diagrams.

    Args:
        hamiltonian_tensor (np.ndarray): The tensor of the Hamiltonian node.
        hamiltonian_node (Node): The Hamiltonian node.
        neighbour_id (str): The identifier of the neighbour node which is the
            root node of the subtree that has already been contracted and is
            saved in the dictionary.
        tensor_leg_to_neighbour (int): The index of the leg of the Hamiltonian tensor
            that points to the neighbour block and is thus to be contracted.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees. The tensors in here can have an arbitrary
            number of legs, but the first leg is the one that is contracted with
            the Hamiltonian tensor.

    Returns:
        np.ndarray: The resulting tensor::

                 _____       out      
                |     |____    
                |     |   4             
                |     |        |1       
                |     |     ___|__      
                |     |    |      |     
                |     |____|   H  |_____
                |     |    |      |     0
                |     |    |______|     
                |     |        |        
                |     |        |2    
                |     |                 
                |     |_____       
                |_____|    3      
                              in

    """
    cached_neighbour_tensor = partial_tree_cache.get_entry(neighbour_id,
                                                           hamiltonian_node.identifier)             
    if tensor_leg_to_neighbour is None:
        tensor_leg_to_neighbour = hamiltonian_node.neighbour_index(neighbour_id)
    return np.tensordot(hamiltonian_tensor, cached_neighbour_tensor,
                        axes=([tensor_leg_to_neighbour],[1]))

def contract_neighbour_block_to_hamiltonian_ignore_one_leg(hamiltonian_tensor: np.ndarray,
                                                           hamiltonian_node: Node,
                                                           neighbour_id: str,
                                                           ignoring_node_id: str,
                                                           partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contract all neighbour blocks to the Hamiltonian tensor.

    Args:
        hamiltonian_tensor (np.ndarray): The tensor of the Hamiltonian node.
        hamiltonian_node (Node): The Hamiltonian node.
        neighbour_id (str): The identifier of the neighbour node which is the
            root node of the subtree that has already been contracted and is
            saved in the dictionary.
        ignoring_node_id (str): The identifier of the node to which the virtual
            leg should not point.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.

    Returns:
        np.ndarray: The resulting tensor::

                 _____       out      
                |     |____    
                |     |   4             
                |     |        |1       
                |     |     ___|__      
                |     |    |      |     
                |     |____|   H  |_____
                |     |    |      |     0
                |     |    |______|     
                |     |        |        
                |     |        |2    
                |     |                 
                |     |_____       
                |_____|    3      
                              in

    """
    tensor_index_to_neighbour = determine_index_with_ignored_leg(hamiltonian_node,
                                                                 neighbour_id,
                                                                 ignoring_node_id)
    return contract_neighbour_block_to_hamiltonian(hamiltonian_tensor,
                                                   hamiltonian_node,
                                                   neighbour_id,
                                                   partial_tree_cache,
                                                   tensor_leg_to_neighbour=tensor_index_to_neighbour)        

def contract_all_but_one_neighbour_block_to_hamiltonian(hamiltonian_tensor: np.ndarray,
                                                        hamiltonian_node: Node,
                                                        next_node_id: str,
                                                        partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contract all neighbour blocks to the Hamiltonian tensor.

    Args:
        hamiltonian_tensor (np.ndarray): The tensor of the Hamiltonian node.
        hamiltonian_node (Node): The Hamiltonian node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual leg points.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.

    Returns:
        np.ndarray: The resulting tensor::

                 _____       out      
                |     |____    
                |     |   4             
                |     |        |1       
                |     |     ___|__      
                |     |    |      |     
                |     |____|   H  |_____
                |     |    |      |     0
                |     |    |______|     
                |     |        |        
                |     |        |2    
                |     |                 
                |     |_____       
                |_____|    3      
                              in

    """
    result_tensor = hamiltonian_tensor
    for neighbour_id in hamiltonian_node.neighbouring_nodes():
        if neighbour_id != next_node_id:
            result_tensor = contract_neighbour_block_to_hamiltonian_ignore_one_leg(result_tensor,
                                                                                   hamiltonian_node,
                                                                                   neighbour_id,
                                                                                   next_node_id,
                                                                                   partial_tree_cache)
    return result_tensor

def contract_all_neighbour_blocks_to_hamiltonian(hamiltonian_tensor: np.ndarray,
                                                 hamiltonian_node: Node,
                                                 partial_tree_cache: PartialTreeCachDict) -> np.ndarray:
    """
    Contract all neighbour blocks to the hamiltonian tensor.

    Args:
        hamiltonian_tensor (np.ndarray): The tensor of the hamiltonian node.
        hamiltonian_node (Node): The hamiltonian node.
        partial_tree_cache (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.

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
    result_tensor = hamiltonian_tensor
    for neighbour_id in hamiltonian_node.neighbouring_nodes():
        # H's neighbours are the same as the leg order, the tensor_leg_to_neighbour
        # is always 0.
        result_tensor = contract_neighbour_block_to_hamiltonian(result_tensor,
                                                                hamiltonian_node,
                                                                neighbour_id,
                                                                partial_tree_cache,
                                                                tensor_leg_to_neighbour=0)
    return result_tensor
