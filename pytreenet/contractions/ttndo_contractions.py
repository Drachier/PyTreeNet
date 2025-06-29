"""
This module contains functions to contrac a TTNDO.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
from re import match
from numpy import tensordot, ndarray, trace, complex128, transpose
from copy import deepcopy

from ..ttno.ttno_class import TreeTensorNetworkOperator
from .tree_cach_dict import PartialTreeCachDict
from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                               get_equivalent_legs)
from .state_state_contraction import contract_any_nodes as contract_any_nodes_state
from .state_operator_contraction import (contract_any_node_environment_but_one as contract_any_nodes_operator,
                                         contract_operator_tensor_ignoring_one_leg)
from ..ttns import TreeTensorNetworkState
from pytreenet.contractions.contraction_util import get_equivalent_legs


if TYPE_CHECKING:
    from ..ttns.ttndo import BINARYTTNDO, SymmetricTTNDO

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


# Binary TTNDO Contractions

def contract_physical_nodes(ttndo: BINARYTTNDO, 
                            bra_suffix: str = "_bra", 
                            ket_suffix: str = "_ket",
                            to_copy: bool = True) -> TreeTensorNetworkState:
    """
    Contracts physical nodes of a binary TTNDO into a regular TTNDO.

    Args:
        ttndo (BINARYTTNDO): The binary TTNDO to contract
        bra_suffix (str): The suffix used for bra nodes (default: "_bra")
        ket_suffix (str): The suffix used for ket nodes (default: "_ket")
        
    Returns:
        TreeTensorNetworkState: A contracted TTNS where physical node has two open legs
    """
    # Create a deep copy to avoid modifying the original
    if to_copy:
        result_ttn = deepcopy(ttndo)
    else:
        result_ttn = ttndo
    
    # Dictionary to track which nodes have been processed
    processed_nodes = {}
    
    # Process nodes in a bottom-up manner to ensure we contract from leaves to root
    linearized_nodes = result_ttn.linearise()

    # First pass: Identify all node pairs that need to be contracted
    node_pairs = []
    for node_id in linearized_nodes:
        # Skip already processed nodes
        if node_id in processed_nodes:
            continue
            
        # Skip nodes that don't have a bra/ket suffix
        if not (node_id.endswith(bra_suffix) or node_id.endswith(ket_suffix)):
            continue
            
        # Get the corresponding lateral node ID
        if node_id.endswith(ket_suffix):
            base_id = node_id[:-len(ket_suffix)]
            lateral_id = base_id + bra_suffix
        else:  # node_id.endswith(bra_suffix)
            base_id = node_id[:-len(bra_suffix)]
            lateral_id = base_id + ket_suffix
        
        # Skip if lateral node doesn't exist or already processed
        if lateral_id not in result_ttn.nodes or lateral_id in processed_nodes:
            continue
            
        # Add to pairs for contraction
        node_pairs.append((node_id, lateral_id, base_id))
        processed_nodes[node_id] = True
        processed_nodes[lateral_id] = True
    
    # Second pass: Determine the correct contraction order (leaves first, then up)
    node_pairs.sort(key=lambda pair: pair[0])
    
    # Third pass: Contract node pairs in the determined order (from leaves to root)
    for _, (node_id, lateral_id, base_id) in enumerate(node_pairs):
        # Check if nodes still exist 
        if node_id not in result_ttn.nodes or lateral_id not in result_ttn.nodes:
            continue
            
        node = result_ttn.nodes[node_id]
        lateral_node = result_ttn.nodes[lateral_id]
        
        # Find the lateral connection
        lateral_connection = False
        
        # Check if nodes are directly connected
        node_neighbors = node.neighbouring_nodes()
        if lateral_id in node_neighbors:
            lateral_connection = True
                
        # If not found, check if the lateral node is connected to the primary node
        if not lateral_connection:
            lateral_node_neighbors = lateral_node.neighbouring_nodes()
            if node_id in lateral_node_neighbors:
                lateral_connection = True
        
        # If directly connected, contract the nodes
        if lateral_connection:
            # Always pass the ket node as the first parameter to contract_nodes
            # This follows the TTNO convention where ket legs come before bra legs.
            try:
                if node_id.endswith(ket_suffix):  # node_id is ket
                    result_ttn.contract_nodes(node_id, lateral_id, new_identifier=base_id)
                else:  # lateral_id is ket
                    result_ttn.contract_nodes(lateral_id, node_id, new_identifier=base_id)
            except Exception:
                raise
        else:
            # Nodes are not directly connected 
            print(f"Warning: Nodes {node_id} and {lateral_id} are not directly connected")

    return result_ttn

def trace_contracted_binary_ttndo(ttndo: 'BINARYTTNDO') -> complex:
    """
    Computes the trace of a contracted physically binary TTNDO.
    The trace is computed by first tracing out the physical legs at each leaf node,
    then contracting up the tree.
    
    Args:
        ttndo (TreeTensorNetworkState): The contracted physically binary TTNDO 
            to compute the trace of.
        
    Returns:
        complex: The trace of the TTNDO.
    """
    # First process all non-root nodes
    nodes_to_process = ttndo.linearise()[:-1]
    for node_id in nodes_to_process:
        if node_id.startswith(ttndo.phys_prefix):
            node, tensor = ttndo[node_id]
            parent_id = node.parent

            tensor_trace = trace(tensor, axis1=-2, axis2=-1)

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
    
    # Now process the root node if it's a physical node
    root_id = ttndo.root_id
    _, root_tensor = ttndo[root_id]

    # Check if root node is a physical node (like in MPS structures)
    if root_id.startswith(ttndo.phys_prefix) or root_tensor.ndim > 1:
        # If the root tensor has physical legs (ndim > 1), trace them out
        if root_tensor.ndim > 1:
            root_tensor = trace(root_tensor, axis1=-2, axis2=-1)
            ttndo.tensors[root_id] = root_tensor
            ttndo.nodes[root_id].link_tensor(root_tensor)

    # Verify that we have a scalar
    if root_tensor.ndim == 0:
        # Scalar result
        return root_tensor.item()
    elif root_tensor.ndim == 1 and root_tensor.shape[0] == 1:
        # [value] format - extract the scalar
        return root_tensor[0]
    else:
        print(f"Root tensor has shape {root_tensor.shape}")
        raise ValueError(f"Root node {root_id} has non-scalar trace")


def contract_ttno_with_ttndo(ttno: TreeTensorNetworkOperator, ttndo: 'BINARYTTNDO') -> TreeTensorNetworkOperator:
    """
    Contract a TTNO with a TTNDO, creating an expanded TTNO.
    
    This contracts each tensor in the TTNO with the corresponding tensor in the TTNDO
    following a specified path order, creating an expanded operator.
    
    Args:
        ttno (TreeTensorNetworkOperator): The tree tensor network operator
        ttndo (TreeTensorNetworkState): The contracted binary TTNDO representing a density matrix
    
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
    
    # Contract the TTNO tensors with the TTNDO tensors node by node
    for node_id in all_nodes:
        # Contract the tensors
        if node_id.startswith(ttndo.phys_prefix):
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

def contract_tensors_ttno_with_ttndo(A: ndarray, B: ndarray) -> ndarray:
    """
    Contract a TTNO tensor with a TTNDO tensor locally.
    
    The last open leg of the TTNO tensor (A) contracts with the second-to-last leg 
    of the TTNDO tensor (B). This preserves the tensor network structure while 
    increasing the effective dimension.
    
    Args:
        A (ndarray): The tensor from the TTNO.
        B (ndarray): The tensor from the TTNDO.
    
    Returns:
        ndarray: The contracted tensor.
    """
    # Contract the last leg of TTNO with the second-to-last leg of TTNDO
    C = tensordot(A, B, axes=((-1,), (-2,)))
    
    # Construct the permutation array for proper leg ordering
    perm = []
    for i in range(((C.ndim-1)//2)):
        perm.append(i)
        perm.append(i + ((A.ndim-1)))
    perm.append((A.ndim-2))
    perm.append((C.ndim-1))
    
    # Apply the permutation
    C = transpose(C, tuple(perm))
    
    # Reshape to merge paired dimensions
    original_shape = C.shape
    new_shape = []
    for i in range(0, len(original_shape)-2, 2):
        new_shape.append(original_shape[i] * original_shape[i + 1])
    new_shape.append(original_shape[-2])
    new_shape.append(original_shape[-1])
    
    C = C.reshape(tuple(new_shape))
    
    return C

def contract_tensors_ttno_with_ttns(A: ndarray, B: ndarray) -> ndarray:
    """
    Contract a TTNO tensor with a TTNS tensor locally.
    
    This function is used specifically for physically binary TTNDOs where virtual nodes
    are represented by regular TTNS tensors. The function contracts the last leg of the 
    TTNO tensor (A) with the last leg of the TTNS tensor (B), then reorders and reshapes
    the resulting tensor to maintain the proper tensor network structure.
    
    Args:
        A (ndarray): The tensor from the TTNO.
        B (ndarray): The tensor from the TTNS (virtual node in physically binary TTNDO).
    
    Returns:
        ndarray: The contracted tensor with properly rearranged legs and merged dimensions.
    """
    C = tensordot( B, A, axes=((-1,), (A.ndim-1,)))
    perm = []
    for i in range(((C.ndim-1)//2)): 
        perm.append(i + ((C.ndim-1)//2) )
        perm.append(i)   
    perm.append((C.ndim-1))
    C = transpose(C, tuple(perm))
    original_shape = C.shape
    new_shape = []
    for i in range(0, len(original_shape)-2, 2):
        new_shape += (original_shape[i] * original_shape[i + 1],)
    new_shape += (original_shape[-1],)
    C = C.reshape(new_shape)   
    return C

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
                raise NotImplementedError()
            ttn3.tensors[node_id] = ttn1_tensor
            ttn3.nodes[node_id].link_tensor(ttn1_tensor)
    return ttn3   

def binary_ttndo_ttno_expectation_value(ttno: TreeTensorNetworkOperator, ttndo: 'BINARYTTNDO') -> complex:
    """
    Compute the expectation value of a TTNO with respect to a physically binary TTNDO.
    This is the trace of the product of the TTNO and the TTNDO:
    <TTNO> = Tr(TTNO @ TTNDO)
    """
    expanded_ttno = contract_ttno_with_ttndo(ttno, ttndo)
    expectation_value = trace_contracted_binary_ttndo(expanded_ttno)
    return complex128(expectation_value)