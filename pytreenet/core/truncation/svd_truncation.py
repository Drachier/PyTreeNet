"""
This file implements the truncation of a tree tensor networks state using the
singular value decomposition (SVD) method. The truncation is performed by
moving the orthogonality center around and truncating the singular values
according to a given threshold.
"""
from uuid import uuid1
from enum import Enum

from ..ttn import TreeTensorNetwork
from ...util.tensor_splitting import (SVDParameters)
from ..leg_specification import LegSpecification
from ...time_evolution.time_evo_util.update_path import find_path

class SVDSiteNumber(Enum):
    """
    The site number to use for SVD truncation.
    """
    ONESITE = 1
    TWOSITE = 2

def svd_truncation(tree: TreeTensorNetwork,
                   params: SVDParameters,
                   site_number: SVDSiteNumber = SVDSiteNumber.ONESITE
                   ) -> TreeTensorNetwork:
    """
    Truncates the tree tensor network using the singular value decomposition
    (SVD) method.

    Args:
        tree (TreeTensorNetwork): The tree tensor network to be truncated.
        params (SVDParameters): The parameters for the SVD truncation.
        site_number (SVDSiteNumber): The site number to use for SVD
            truncation. Defaults to ONESITE.

    Returns:
        TreeTensorNetwork: The modified truncated tree tensor network.
            It will be in site canonical form with respect to the root.
    """
    # Find the update path for the truncation.
    update_path = find_path(tree)
    tree.canonical_form(update_path[0])
    # Move the orthogonality center along the path and truncate the tensors.
    for node_id in update_path[:-1]: # Root not needed.
        tree.move_orthogonalization_center(node_id)
        if site_number is SVDSiteNumber.ONESITE:
            compress_parent_leg(node_id, tree, params)
        elif site_number is SVDSiteNumber.TWOSITE:
            contract_and_split_with_parent(node_id, tree, params)
        else:
            raise ValueError(f"Unknown site number: {site_number}")
    return tree

def compress_parent_leg(node_id: str,
                        tree: TreeTensorNetwork,
                        params: SVDParameters):
    """
    Compress the leg between a node and its parent via SVD.

    Args:
        node_id (str): The identifier of the node whose parent leg is to be
            compressed.
        tree (TreeTensorNetwork): The tree tensor network containing the node.
        params (SVDParameters): The parameters for the SVD truncation.
    
    """
    parent_id = get_parent_id(node_id, tree)
    node = tree.nodes[node_id]
    child_legs = LegSpecification.all_but_parent(node)
    parent_legs = LegSpecification.only_parent(node)
    matrix_id = str(uuid1())
    tree.split_node_svd(node_id,
                        child_legs,
                        parent_legs,
                        u_identifier=node_id,
                        v_identifier=matrix_id,
                        svd_params=params)
    tree.contract_nodes(matrix_id, parent_id,
                        new_identifier=parent_id)
    tree.orthogonality_center_id = parent_id

def contract_and_split_with_parent(node_id: str,
                                   tree: TreeTensorNetwork,
                                   params: SVDParameters):
    """
    Contract a node with its parent and split the resulting node via SVD.

    The parent node will absorb the remaining singular values and thus becomes
    the new orthogonality center.

    Args:
        node_id (str): The identifier of the node to be contracted and split.
        tree (TreeTensorNetwork): The tree tensor network containing the node.
        params (SVDParameters): The parameters for the SVD truncation.
    
    """
    parent_id = get_parent_id(node_id, tree)
    child_legs, parent_legs = tree.legs_before_combination(node_id, parent_id)
    contr_id = str(uuid1())
    tree.contract_nodes(node_id, parent_id,
                        new_identifier=contr_id)
    tree.split_node_svd(contr_id, child_legs, parent_legs,
                        node_id, parent_id, params)
    tree.orthogonality_center_id = get_parent_id(node_id, tree)

def get_parent_id(node_id: str, tree: TreeTensorNetwork) -> str:
    """
    Get the identifier of the parent of a node in a tree tensor network.

    Args:
        node_id (str): The identifier of the node.
        tree (TreeTensorNetwork): The tree tensor network containing the node.

    Returns:
        str: The identifier of the parent of the node.
    """
    return tree.nodes[node_id].parent
