"""
This file implements the truncation of a tree tensor networks state using the
singular value decomposition (SVD) method. The truncation is performed by
moving the orthogonality center around and truncating the singular values
according to a given threshold.
"""
from uuid import uuid1

from ..ttn import TreeTensorNetwork
from ...util.tensor_splitting import (SVDParameters)

def svd_truncation(tree: TreeTensorNetwork, params: SVDParameters) -> TreeTensorNetwork:
    """
    Truncates the tree tensor network using the singular value decomposition
    (SVD) method.

    Args:
        tree (TreeTensorNetwork): The tree tensor network to be truncated.
        params (SVDParameters): The parameters for the SVD truncation.

    Returns:
        TreeTensorNetwork: The modified truncated tree tensor network.
    """
    # Find the update path for the truncation.
    update_path = tree.linearise()
    tree.canonical_form(update_path[0])
    # Move the orthogonality center along the path and truncate the tensors.
    for node_id in update_path[:-1]:
        tree.move_orthogonalization_center(node_id)
        contract_and_split_with_parent(node_id, tree, params)
    return tree

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
