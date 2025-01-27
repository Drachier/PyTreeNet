"""
This module contains functions to contrac a TTNDO.
"""
from __future__ import annotations
from typing import List
from re import match

from numpy import tensordot, ndarray

from ..ttno.ttno_class import TreeTensorNetworkOperator
from .tree_cach_dict import PartialTreeCachDict
from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                               get_equivalent_legs)
from .state_state_contraction import contract_any_nodes as contract_any_nodes_state
from .state_operator_contraction import (contract_any_node_environment_but_one as contract_any_nodes_operator,
                                         contract_operator_tensor_ignoring_one_leg)

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
    final_ket_id = contraction_order[-1]
    final_block = block_cache.get_entry(final_ket_id,
                                        ttndo.root_id)
    return _contract_final_block(ttndo, final_block)

def ttndo_ttno_expectation_value(ttndo: SymmetricTTNDO,
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

def _contract_final_block(ttndo: SymmetricTTNDO,
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

def _contract_ttno_root(ttndo: SymmetricTTNDO,
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
