"""
This file implements the truncation of a tree tensor networks state using the
singular value decomposition (SVD) method. The truncation is performed by
moving the orthogonality center around and truncating the singular values
according to a given threshold.
"""
from copy import copy

from ...ttns.ttns import TreeTensorNetworkState
from ...util.tensor_splitting import SVDParameters
from ..canonical_form import split_svd_contract_sv_to_neighbour
from .truncation_util import (get_truncation_projector,
                                insert_child_projection_operator_and_conjugate,
                                contract_projectors,
                                move_orth_for_path)

def recursive_downward_truncation(ttn: TreeTensorNetworkState,
                                  svd_params: SVDParameters) -> TreeTensorNetworkState:
    """
    Truncates the tree tensor network using the recursive truncation method.

    Args:
        ttn (TreeTensorNetworkState): The tree tensor network to be truncated.
        svd_params (SVDParameters): The parameters for the SVD truncation.
    Returns:
        TreeTensorNetworkState: The modified truncated tree tensor network.
    """
    # Ensure the orthogonality center is at the root node
    root_id = ttn.root_id
    if root_id != ttn.orthogonality_center_id or ttn.orthogonality_center_id is None:
        ttn.canonical_form(root_id)

    svd_truncate_children(root_id, ttn, svd_params)

    # Reset orthogonality center after truncation
    ttn.orthogonality_center_id = None

    return ttn

def svd_truncate_children(node_id: str,
                            ttn: TreeTensorNetworkState,
                            svd_params: SVDParameters):
    """
    Truncates the children of the node with the given id.
    Args:
        node_id (str): The id of the node to truncate.
        ttn (TreeTensorNetworkState): The tree tensor network state.
        svd_params (SVDParameters): The parameters for the SVD truncation.
    """
    node = ttn.nodes[node_id]
    orig_children = copy(node.children)
    for child_id in orig_children:
        projector, _ = get_truncation_projector(node,
                                            ttn.tensors[node_id],
                                            child_id,
                                            svd_params)
        insert_child_projection_operator_and_conjugate(child_id,
                                                    node_id,
                                                    projector,
                                                    ttn)

    contract_projectors(ttn, node_id, contract_parent = False)

    # This makes the former children, the new children
    for child_id in orig_children:
        ttn.move_orthogonalization_center(child_id)
        svd_truncate_children(child_id, ttn, svd_params)

def sweeping_onward_truncation(ttn,
                            update_path,
                            orth_path,
                            svd_params):
    """
    - Truncates the tree tensor network by sweeping through the update path, 
    - The orthogonalization center is moved to the target node that is being truncated.
    Args:
        ttn (TreeTensorNetworkState): The tree tensor network state to be truncated.
        update_path (list): The path to be followed for truncation.
        orth_path (list): The orthogonalization path to move the orthogonality center.
        svd_params (SVDParameters): The parameters for the SVD truncation.
    """
    if ttn.orthogonality_center_id is None:
        ttn.canonical_form(update_path[0])
    elif ttn.orthogonality_center_id != update_path[0]:
        ttn.move_orthogonalization_center(update_path[0])
    
    for update_index, node_id in enumerate(update_path):
        if update_index == 0:
            next_node_id = orth_path[update_index][0]
            split_svd_contract_sv_to_neighbour(ttn, node_id, next_node_id, svd_params)
            ttn.orthogonality_center_id = next_node_id
        elif update_index < len(orth_path):
            next_node_id = orth_path[update_index][0]
            current_orth_path = orth_path[update_index-1]
            move_orth_for_path(ttn, current_orth_path)
            split_svd_contract_sv_to_neighbour(ttn, node_id, next_node_id, svd_params)
            ttn.orthogonality_center_id = next_node_id
    assert ttn.orthogonality_center_id == update_path[-1]