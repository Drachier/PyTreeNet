"""
This module contains functions to contrac a TTNDO.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
from re import match
from copy import deepcopy

from numpy import tensordot, ndarray, trace
import numpy as np

from ..ttno.ttno_class import TreeTensorNetworkOperator
from .tree_cach_dict import PartialTreeCachDict
from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                               get_equivalent_legs,
                               determine_index_with_ignored_leg)
from .state_state_contraction import contract_any_nodes as contract_any_nodes_state
from .state_operator_contraction import (contract_any_node_environment_but_one as contract_any_nodes_operator,
                                         contract_operator_tensor_ignoring_one_leg)

# Move the class imports inside TYPE_CHECKING block to avoid circular imports
if TYPE_CHECKING:
    from ..ttns.ttndo import IntertwinedTTNDO, SymmetricTTNDO

def ttndo_contraction_order(ttndo: 'SymmetricTTNDO') -> List[str]:
    """
    Returns the contraction order of a TTNDO.

    In this case the bra and ket nodes are not contained sperately, since they
    are contracted with each other. The nodes with the ket suffix are returned.

    Args:
        ttndo (SymmetricTTNDO): The TTNDO to determine the contraction order of.

    Returns:
        List[str]: The contraction order of the TTNDO, excluding the root node.
    
    """
    all_node_ids = ttndo.linearise()
    # Filter out all bra nodes
    bra_node_ids = [node_id for node_id in all_node_ids
                    if match(r".*"+ttndo.ket_suffix, node_id)]
    return bra_node_ids

def trace_symmetric_ttndo(ttndo: 'SymmetricTTNDO') -> complex:
    """
    Computes the trace of a TTNDO.

    Args:
        ttndo (SymmetricTTNDO): The TTNDO to compute the trace of.

    Returns:
        complex: The trace of the TTNDO.
    
    """
    if ttndo.root_id is None:
        return 0
    if len(ttndo.nodes) == 1:
        errstr = "The TTNDO has only a root node. Thus, the trace is not well-defined!"
        raise ValueError(errstr)
    contraction_order = ttndo_contraction_order(ttndo) # ket nodes
    block_cache = PartialTreeCachDict()
    id_trafo = ttndo.ket_to_bra_id
    for ket_id in contraction_order:
        ket_node, ket_tensor = ttndo[ket_id]
        next_ket_id = ket_node.parent
        bra_node, bra_tensor = ttndo[id_trafo(ket_id)]
        block = contract_any_nodes_state(next_ket_id,
                                         ket_node,
                                         bra_node,
                                         ket_tensor,
                                         bra_tensor,
                                         block_cache,
                                         id_trafo=id_trafo)
        block_cache.add_entry(ket_id, next_ket_id, block)
        # The children blocks are not needed anymore
        ket_children = ket_node.children
        for child_id in ket_children:
            block_cache.delete_entry(child_id, ket_id)
    # Now we need to contract the root/symmetry center to the block
    final_ket_id = contraction_order[-1]
    final_block = block_cache.get_entry(final_ket_id,
                                        ttndo.root_id)
    return _contract_final_block(ttndo, final_block)

def symmetric_ttndo_ttno_expectation_value(ttndo: 'SymmetricTTNDO',
                                 ttno: TreeTensorNetworkOperator
                                 ) -> complex:
    """
    Computes the expectation value of a TTNDO with respect to a TTNO.

    ..math::
        <TTNO> = Tr(TTNO @ TTNDO)

    Args:
        ttndo (SymmetricTTNDO): The TTNDO to compute the expectation value of.
        ttno (TreeTensorNetworkOperator): The TTNO to compute the expectation
            value with.
    
    Returns:
        complex: The expectation value of the TTNDO with respect to the TTNO.

    """
    if ttndo.root_id is None and ttno.root_id is None:
        return 0
    # ket nodes, but the last, as the ttno node will have on eless neighbour
    contraction_order = ttndo_contraction_order(ttndo)[:-1]
    block_cache = PartialTreeCachDict()
    id_bra_trafo = ttndo.ket_to_bra_id
    id_op_trafo = ttndo.reverse_ket_id
    for ket_id in contraction_order:
        ket_node, ket_tensor = ttndo[ket_id]
        next_ket_id = ket_node.parent
        bra_node, bra_tensor = ttndo[id_bra_trafo(ket_id)]
        op_node, op_tensor = ttno[id_op_trafo(ket_id)]
        block = contract_any_nodes_operator(next_ket_id,
                                            ket_node, ket_tensor,
                                            op_node, op_tensor,
                                            block_cache,
                                            bra_node=bra_node,
                                            bra_tensor=bra_tensor,
                                            id_trafo_op=id_op_trafo,
                                            id_trafo_bra=id_bra_trafo)
        block_cache.add_entry(ket_id, next_ket_id, block)
        # The children blocks are not needed anymore
        ket_children = ket_node.children
        for child_id in ket_children:
            block_cache.delete_entry(child_id, ket_id)
    # Now we need to contract the root of the ttno with the respective bra and
    # ket nodes and neighbour blocks.
    # This is required, as the ttno root will have one less neighbour than the
    # respective bra and ket nodes.
    final_block = _contract_ttno_root(ttndo, ttno, block_cache)
    # Now we need to contract the final block to the root of the TTNDO
    return _contract_final_block(ttndo, final_block)

def _contract_final_block(ttndo: 'SymmetricTTNDO',
                          final_block: ndarray
                          ) -> complex:
    """
    Contract the final environment block to the root of the TTNDO.

    Args:
        ttndo (SymmetricTTNDO): The TTNDO to contract the final block with.
        final_block (ndarray): The final block to contract with the TTNDO root.
    
    Returns:
        complex: The contracted final block.

    """
    root_node, root_tensor = ttndo[ttndo.root_id]
    assert root_node.nneighbours() == 2, "The root node has to have two neighbours!"
    final_ket_id = [child_id for child_id in root_node.children
                    if child_id.endswith(ttndo.ket_suffix)][0]
    neigbour_ids = [final_ket_id, ttndo.ket_to_bra_id(final_ket_id)]
    root_legs = [root_node.neighbour_index(neigbour_id)
                 for neigbour_id in neigbour_ids]
    block_legs = [0,1] # ket_leg, bra_leg
    contraction_result = tensordot(root_tensor,
                             final_block,
                             axes=(root_legs, block_legs))
    return contraction_result[0]

def _contract_ttno_root(ttndo: 'SymmetricTTNDO',
                        ttno: TreeTensorNetworkOperator,
                        block_cache: PartialTreeCachDict
                        ) -> ndarray:
    """
    Contract the root of the TTNO with the bra and ket node of the TTNDO.

    Args:
        ttndo (SymmetricTTNDO): The TTNDO to contract the root of the TTNO
            with.
        ttno (TreeTensorNetworkOperator): The TTNO to contract the root with.
        block_cache (PartialTreeCachDict): The cache to store the neighbour
            blocks in
    
    Returns:
        ndarray: The contracted block.

    """
    root_id = ttno.root_id
    root_node, root_tensor = ttno[root_id]
    ket_node, ket_tensor = ttndo[ttndo.ket_id(root_id)]
    bra_node, bra_tensor = ttndo[ttndo.bra_id(root_id)]
    if len(ttno.nodes) == 1:
        # This corresponds to contracting a leaf node.
        assert len(block_cache) == 0, "The block cache has to be empty!"
        return _single_site_contraction(ket_tensor, root_tensor, bra_tensor)
    ttndo_root_id = ttndo.root_id
    ketblock_tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                                  ket_node,
                                                                  ttndo_root_id,
                                                                  block_cache)
    # This works, as for this step we can just ignore the missing leg
    ketopblock_tensor = contract_operator_tensor_ignoring_one_leg(ketblock_tensor,
                                                                  ket_node,
                                                                  root_tensor,
                                                                  root_node,
                                                                  ttndo_root_id,
                                                                  id_trafo=ttndo.reverse_ket_id)
    # However, due to the missing leg, the tensor leg order is now slightly different
    num_neighbours = ket_node.nneighbours()
    # The 0th leg is the open leg to the ttndo root, while the last leg is the
    # physical leg originating from the ttno root
    legs_tensor = list(range(1,num_neighbours+1))
    _, legs_bra_tensor = get_equivalent_legs(ket_node,
                                            bra_node,
                                            ignore_legs=[ttndo_root_id],
                                            id_trafo=ttndo.ket_to_bra_id)
    # Adding the physical leg to be contracted.
    legs_bra_tensor.append(bra_node.nneighbours())
    final_block = tensordot(ketopblock_tensor, bra_tensor,
                        axes=(legs_tensor, legs_bra_tensor))
    return final_block

def _single_site_contraction(ket_tensor: ndarray,
                             root_tensor: ndarray,
                             bra_tensor: ndarray
                             ) -> ndarray:
    """
    Contract the final environment block if the TTNO has only a single site.

    Args:
        ket_tensor (ndarray): The ket tensor.
        root_tensor (ndarray): The root tensor.
        bra_tensor (ndarray): The bra tensor.

    Returns:
        ndarray: The contracted block.

    """
    assert ket_tensor.ndim == 2, "The ket tensor has to be a matrix!"
    assert root_tensor.ndim == 2, "The root tensor has to be a matrix!"
    assert bra_tensor.ndim == 2, "The bra tensor has to be a matrix!"
    # The ket tensor needs to be transposed, as its physical leg is leg 1,
    # while in matrix multiplication the phyisical leg needs to be the output
    # so the 0th leg.
    block = bra_tensor @ root_tensor @ ket_tensor.T
    # The transpose is needed, as we want the ket leg to be the first leg
    return block.T


# fully intertwined TTNDO tracing
def trace_contracted_fully_intertwined_ttndo(ttndo: 'IntertwinedTTNDO') -> complex:
    """
    Computes the trace of a contracted intertwined TTNDO.
    
    This function is used for TTNDOs that have been processed through 
    contract_intertwined_ttndo() and have each tensor with two open physical
    legs representing density matrix indices.
    
    Args:
        ttndo (TreeTensorNetworkState): The contracted intertwined TTNDO to compute the trace of.
        
    Returns:
        complex: The trace of the TTNDO.
    """
    
    # Create a cache dictionary to store intermediate results
    dictionary = PartialTreeCachDict()
    
    # Process in bottom-up order using linearise
    nodes_to_process = ttndo.linearise()
    
    # Process all nodes except root
    for node_id in nodes_to_process[:-1]:
        node, tensor = ttndo[node_id]
        parent_id = node.parent
        
        traced_tensor = trace_subtrees_using_dictionary(parent_id, node, tensor, dictionary)
        dictionary.add_entry(node_id, parent_id, traced_tensor)
        
        # Delete child entries that are no longer needed
        for child_id in node.children:
            dictionary.delete_entry(child_id, node_id)
    
    # Process the root node
    root_id = ttndo.root_id
    root_node, root_tensor = ttndo[root_id]
    if not root_node.children or not any(dictionary.contains(child_id, root_id) for child_id in root_node.children):
        result = trace(root_tensor, axis1=-2, axis2=-1)
        return complex(result if np.isscalar(result) else result.item())
    
    # Process root with its children
    final_result = trace_subtrees_using_dictionary(None, root_node, root_tensor, dictionary)
    
    # Convert the result to a complex number
    if np.isscalar(final_result):
        return complex(final_result)
    else:
        raise ValueError(f"Final trace result is not a scalar")

def trace_subtrees_using_dictionary(parent_id, node, tensor, dictionary):
    """
    Traces a node with all its subtrees, retaining only one leg connection to parent.

    This function contracts child subtrees first, then traces the physical legs.

    Args:
        parent_id: The ID of the parent node (can be None for root)
        node: The current node being processed
        tensor: The tensor for the current node
        dictionary: Dictionary containing processed subtrees

    Returns:
        np.ndarray: Tensor with all child contractions performed and traced
    """
    node_id = node.identifier
    result_tensor = np.copy(tensor)

    children_to_process = []
    child_ids = node.children
    for neighbour_id in child_ids:
        current_index = node.neighbour_index(neighbour_id)
        children_to_process.append((neighbour_id, current_index))

    children_to_process.sort(key=lambda x: x[1], reverse=True)

    for neighbour_id, tensor_index in children_to_process:
        cached_neighbour_tensor = dictionary.get_entry(neighbour_id, node_id)
        result_tensor = np.tensordot(result_tensor,
                                    cached_neighbour_tensor,
                                    axes=([tensor_index], [0]))
        
    result_tensor = trace(result_tensor, axis1=-2, axis2=-1)

    if parent_id is None:
        if np.isscalar(result_tensor):
            return result_tensor
        else:
            raise ValueError(f"Error: Root node {node_id} trace resulted in non-scalar shape: {getattr(result_tensor, 'shape', 'N/A')}")
    return result_tensor

def contract_root_with_environment(ttndo: 'IntertwinedTTNDO', dictionary: PartialTreeCachDict) -> complex:
    """
    Contracts the root node with its environment in a contracted intertwined TTNDO.
    
    Args:
        ttndo (TreeTensorNetworkState): The contracted intertwined TTNDO.
        dictionary (PartialTreeCachDict): The cache dictionary storing results.
        
    Returns:
        complex: The trace of the entire TTNDO.
    """
    node_id = ttndo.root_id
    node, tensor = ttndo[node_id]
    result_tensor = tensor
    
    # Contract with all children's traced results
    for neighbour_id in node.neighbouring_nodes():
            
        cached_neighbour_tensor = dictionary.get_entry(neighbour_id, node_id)
        tensor_leg_to_neighbour = node.neighbour_index(neighbour_id)
        
        # Check tensor dimensions before contracting
        if tensor_leg_to_neighbour >= result_tensor.ndim:
            continue
            
        neighbour_dim = result_tensor.shape[tensor_leg_to_neighbour]
        if cached_neighbour_tensor.shape[0] != neighbour_dim:
            
            # Create a reshaped neighbor tensor if needed
            min_dim = min(neighbour_dim, cached_neighbour_tensor.shape[0])
            new_shape = list(cached_neighbour_tensor.shape)
            new_shape[0] = neighbour_dim
            new_tensor = np.zeros(tuple(new_shape), dtype=cached_neighbour_tensor.dtype)
            
            # Copy the data for the common dimensions
            slices = tuple([slice(0, min_dim)] + [slice(None) for _ in range(cached_neighbour_tensor.ndim-1)])
            if cached_neighbour_tensor.ndim == 1:
                # Handle the 1D case separately
                new_tensor[:min_dim] = cached_neighbour_tensor[:min_dim]
            else:
                new_tensor[slices] = cached_neighbour_tensor[slices[:cached_neighbour_tensor.ndim]]
            
            cached_neighbour_tensor = new_tensor
        
        # Contract with compatible dimensions
        result_tensor = tensordot(result_tensor, cached_neighbour_tensor, axes=([tensor_leg_to_neighbour], [0]))
    
    # Trace over the physical legs (last two dimensions)
    return complex(trace(result_tensor, axis1=-2, axis2=-1))

# physical intertwined TTNDO tracing
def trace_contracted_physically_intertwined_ttndo(ttndo: 'IntertwinedTTNDO') -> complex:
    """
    Computes the trace of a contracted physically intertwined TTNDO.
    The trace is computed by first tracing out the physical legs at each leaf node,
    then contracting up the tree, and finally taking the inner product with the 
    conjugate of the remaining virtual nodes.
    
    Args:
        ttndo (TreeTensorNetworkState): The contracted physically intertwined TTNDO 
            to compute the trace of.
        
    Returns:
        complex: The trace of the TTNDO.
    """
    nodes_to_process = ttndo.linearise()[:-1]
    for node_id in nodes_to_process :
        # Only trace out ical nodes
        if node_id.startswith("qubit"):
            node, tensor = ttndo[node_id]
            parent_id = node.parent

            tensor_trace = np.trace(tensor, axis1=-2, axis2=-1)

            ttndo.tensors[node_id] = tensor_trace
            ttndo.nodes[node_id].link_tensor(tensor_trace)

            ttndo.contract_nodes(parent_id, node_id, parent_id)
        else:    
            node, tensor = ttndo[node_id]
            parent_id = node.parent

            new_shape = tensor.shape[:-1]
            tensor = tensor.reshape(new_shape)
            ttndo.tensors[node_id] = tensor
            ttndo.nodes[node_id].link_tensor(tensor)

            ttndo.contract_nodes(parent_id, node_id, parent_id)

    if not ttndo.tensors[ttndo.root_id].ndim == 1:
        raise ValueError(f"Root node {ttndo.root_id} has non-scalar trace")
    
    return ttndo.tensors[ttndo.root_id][0]

# Fully intertwined TTNDO expectation value
def fully_intertwined_ttndo_ttno_expectation_value(ttno: TreeTensorNetworkOperator, ttndo: 'IntertwinedTTNDO') -> complex:
    """
    Compute the expectation value of a TTNO with respect to a contracted intertwined TTNDO.
    
    This is the trace of the product of the TTNO and the TTNDO:
    <TTNO> = Tr(TTNO @ TTNDO)
    
    The calculation is done in two steps:
    1. Contract the TTNO with the TTNDO locally, creating an expanded TTNO 
    2. Calculate the trace of the resulting expanded TTNO
    
    Args:
        ttno: The tree tensor network operator
        ttndo: The contracted intertwined TTNDO representing a density matrix
    
    Returns:
        complex: The expectation value <TTNO>
    """
    # Step 1: Contract the TTNO with the TTNDO
    expanded_ttno = contract_ttno_with_ttndo(ttno, ttndo)
    
    # Step 2: Calculate the trace of the resulting expanded TTNO
    expectation_value = trace_contracted_fully_intertwined_ttndo(expanded_ttno)
    return np.complex128(expectation_value)

def contract_tensors_ttno_with_ttndo(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Contract a TTNO tensor with a TTNDO tensor locally.
    
    The last open leg of the TTNO tensor (A) contracts with the second-to-last leg 
    of the TTNDO tensor (B). This preserves the tensor network structure while 
    increasing the effective dimension.
    
    Args:
        A (np.ndarray): The tensor from the TTNO.
        B (np.ndarray): The tensor from the TTNDO.
    
    Returns:
        np.ndarray: The contracted tensor.
    """
    # Contract the last leg of TTNO with the second-to-last leg of TTNDO
    C = np.tensordot(A, B, axes=((-1,), (-2,)))
    
    # Construct the permutation array for proper leg ordering
    perm = []
    for i in range(((C.ndim-1)//2)):
        perm.append(i)
        perm.append(i + ((A.ndim-1)))
    perm.append((A.ndim-2))
    perm.append((C.ndim-1))
    
    # Apply the permutation
    C = np.transpose(C, tuple(perm))
    
    # Reshape to merge paired dimensions
    original_shape = C.shape
    new_shape = []
    for i in range(0, len(original_shape)-2, 2):
        new_shape.append(original_shape[i] * original_shape[i + 1])
    new_shape.append(original_shape[-2])
    new_shape.append(original_shape[-1])
    
    C = C.reshape(tuple(new_shape))
    
    return C

def contract_tensors_ttno_with_ttns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Contract a TTNO tensor with a TTNS tensor locally.
    
    This function is used specifically for physically intertwined TTNDOs where virtual nodes
    are represented by regular TTNS tensors. The function contracts the last leg of the 
    TTNO tensor (A) with the last leg of the TTNS tensor (B), then reorders and reshapes
    the resulting tensor to maintain the proper tensor network structure.
    
    Args:
        A (np.ndarray): The tensor from the TTNO.
        B (np.ndarray): The tensor from the TTNS (virtual node in physically intertwined TTNDO).
    
    Returns:
        np.ndarray: The contracted tensor with properly rearranged legs and merged dimensions.
    """
    C = np.tensordot( B, A, axes=((-1,), (A.ndim-1,)))
    perm = []
    for i in range(((C.ndim-1)//2)): 
        perm.append(i + ((C.ndim-1)//2) )
        perm.append(i)   
    perm.append((C.ndim-1))
    C = np.transpose(C, tuple(perm))
    original_shape = C.shape
    new_shape = []
    for i in range(0, len(original_shape)-2, 2):
        new_shape += (original_shape[i] * original_shape[i + 1],)
    new_shape += (original_shape[-1],)
    C = C.reshape(new_shape)   
    return C

from pytreenet.contractions.contraction_util import get_equivalent_legs
def cehck_two_ttn_compatibility(ttn1, ttn2):
    for nodes in ttn1.nodes:
        legs = get_equivalent_legs(ttn1.nodes[nodes], ttn2.nodes[nodes])
        assert legs[0] == legs[1]
        
def adjust_ttn1_structure_to_ttn2(ttn1, ttn2):
    try:
        cehck_two_ttn_compatibility(ttn1, ttn2)
        return ttn1
    except AssertionError: 
        ttn3 = deepcopy(ttn2)
        for node_id in ttn3.nodes:
            ttn1_neighbours = ttn1.nodes[node_id].neighbouring_nodes()
            element_map = {elem: i for i, elem in enumerate(ttn1_neighbours)}
            ttn1_neighbours = ttn2.nodes[node_id].neighbouring_nodes()
            permutation = tuple(element_map[elem] for elem in ttn1_neighbours)
            nneighbours = ttn2.nodes[node_id].nneighbours()
            if len(ttn1.nodes[node_id].open_legs) == 1 :
               ttn1_tensor = ttn1.tensors[node_id].transpose(permutation + (nneighbours,))
            elif len(ttn1.nodes[node_id].open_legs) == 2 :
                ttn1_tensor = ttn1.tensors[node_id].transpose(permutation + (nneighbours,nneighbours+1))
            else:
                # Handle cases with unexpected number of open legs
                raise ValueError(f"Node {node_id} in ttn1 has an unexpected number of open legs: {len(ttn1.nodes[node_id].open_legs)}")
            ttn3.tensors[node_id] = ttn1_tensor
            ttn3.nodes[node_id].link_tensor(ttn1_tensor)
    return ttn3   

def contract_ttno_with_ttndo(ttno: TreeTensorNetworkOperator, ttndo: 'IntertwinedTTNDO') -> TreeTensorNetworkOperator:
    """
    Contract a TTNO with a TTNDO, creating an expanded TTNO.
    
    This contracts each tensor in the TTNO with the corresponding tensor in the TTNDO
    following a specified path order, creating an expanded operator.
    
    Args:
        ttno (TreeTensorNetworkOperator): The tree tensor network operator
        ttndo (TreeTensorNetworkState): The contracted intertwined TTNDO representing a density matrix
    
    Returns:
        TreeTensorNetworkOperator: The expanded TTNO after contraction
        
    Raises:
        ValueError: If the TTNO and TTNDO do not have the same structure.
    """    
    # Create a copy of the TTNO to modify
    ttndo_copy = deepcopy(ttndo)
    ttno = adjust_ttn1_structure_to_ttn2(ttno, ttndo)
    # Get the path for contractions using TDVPUpdatePathFinder
    all_nodes = ttndo.linearise()
    
    if ttndo.form == "full":
        # Contract the TTNO tensors with the TTNDO tensors node by node
        for node_id in all_nodes:
            # Contract the tensors
            ttno_ttndo_tensor = contract_tensors_ttno_with_ttndo(
                ttno.tensors[node_id], 
                ttndo.tensors[node_id])
            # Update the expanded TTNO
            ttndo_copy.tensors[node_id] = ttno_ttndo_tensor
            ttndo_copy.nodes[node_id].link_tensor(ttno_ttndo_tensor)
    elif ttndo.form == "physical":
        # Contract the TTNO tensors with the TTNDO tensors node by node
        for node_id in all_nodes:
            # Contract the tensors
            if node_id.startswith("qubit"):
                ttno_ttndo_tensor = contract_tensors_ttno_with_ttndo(
                    ttno.tensors[node_id], 
                    ttndo.tensors[node_id])
                
                ttndo_copy.tensors[node_id] = ttno_ttndo_tensor
                ttndo_copy.nodes[node_id].link_tensor(ttno_ttndo_tensor)
            else:     
                ttno_ttndo_tensor = contract_tensors_ttno_with_ttns(
                    ttno.tensors[node_id], 
                    ttndo.tensors[node_id])
                
                ttndo_copy.tensors[node_id] = ttno_ttndo_tensor
                ttndo_copy.nodes[node_id].link_tensor(ttno_ttndo_tensor)
    return ttndo_copy

# physically intertwined TTNDO expectation value
def physically_intertwined_ttndo_ttno_expectation_value(ttno: TreeTensorNetworkOperator, ttndo: 'IntertwinedTTNDO') -> complex:

    """
    Compute the expectation value of a TTNO with respect to a physically intertwined TTNDO.
    
    This is the trace of the product of the TTNO and the TTNDO:
    <TTNO> = Tr(TTNO @ TTNDO)
    
    """
    expanded_ttno = contract_ttno_with_ttndo(ttno, ttndo)
    expectation_value = trace_contracted_physically_intertwined_ttndo(expanded_ttno)
    return np.complex128(expectation_value)