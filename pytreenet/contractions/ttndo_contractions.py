"""
This module contains functions to contrac a TTNDO.
"""
from __future__ import annotations
from typing import List
from re import match

from numpy import tensordot

from .tree_cach_dict import PartialTreeCachDict
from .state_state_contraction import contract_any_nodes as contract_any_nodes_state

def ttndo_contraction_order(ttndo: SymmetricTTNDO) -> List[str]:
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

def trace_ttndo(ttndo: SymmetricTTNDO) -> complex:
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
    root_node, root_tensor = ttndo[ttndo.root_id]
    assert root_node.nneighbours() == 2, "The root node has to have two neighbours!"
    final_ket_id = contraction_order[-1]
    final_block = block_cache.get_entry(final_ket_id,
                                        ttndo.root_id)
    neigbour_ids = [final_ket_id, id_trafo(final_ket_id)]
    root_legs = [root_node.neighbour_index(neigbour_id)
                 for neigbour_id in neigbour_ids]
    block_legs = [0,1] # ket_leg, bra_leg
    trace_result = tensordot(root_tensor,
                             final_block,
                             axes=(root_legs, block_legs))
    return trace_result[0]
