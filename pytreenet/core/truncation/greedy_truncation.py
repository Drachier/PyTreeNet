"""
This file implements the truncation of a tree tensor networks state using the
singular value decomposition (SVD) method followed by a greedy optimization 
of the bond dimensions of each node. The truncation is performed by
moving the orthogonality center around and truncating the singular values.
"""

from copy import copy
from warnings import warn
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.util.tensor_splitting import SVDParameters

from pytreenet.core.truncation.truncation_util import (get_truncation_projector,
                                                       insert_parent_projection_operator_and_conjugate,
                                                       insert_child_projection_operator_and_conjugate,
                                                       calculate_node_dimension_product,
                                                       optimize_bond_dimensions,
                                                       contract_projectors)

def recursive_greedy_truncation(tree: TreeTensorNetworkState,
                                svd_params: SVDParameters,
                                **kwargs) -> TreeTensorNetworkState:
    """
    Truncates the tree tensor network using the recursive truncation method.

    Args:
        tree (TreeTensorNetworkState): The tree tensor network to be truncated.
        svd_params (SVDParameters): The parameters for the SVD truncation.
        **kwargs: Additional keyword arguments to be passed to the SVD splitting functions.
    Returns:
        TreeTensorNetworkState: The modified truncated tree tensor network.
    """
    # Ensure the orthogonality center is at the root node
    root_id = tree.root_id
    if root_id != tree.orthogonality_center_id or tree.orthogonality_center_id is None:
        tree.canonical_form(root_id, **kwargs)

    greedy_truncate_neighbours(root_id, tree, svd_params)

    tree.orthogonality_center_id = None

def greedy_truncate_neighbours(node_id: str,
                                tree: TreeTensorNetworkState,
                                svd_params: SVDParameters):
    """
    Truncates the specified node along all neghbours recursively down to leaves.

    Unlike svd_truncate_children, this function processes both child and parent connections,
    but doesn't recursively process child nodes.

    Args:
        node_id (str): The id of the node to truncate.
        tree (TreeTensorNetworkState): The tree tensor network to truncate.
        svd_params (SVDParameters): The parameters for the SVD truncation.
    """
    assert tree.orthogonality_center_id == node_id, \
        "The orthogonality center must be at node_id start the truncation."

    include_parent = False
    greedy_opt = False

    svd_params.max_bond_dim = float('inf')

    node = tree.nodes[node_id]

    orig_children = copy(node.children)
    orig_neighbors = list(orig_children)
    parent_id = node.parent
    is_root = parent_id is None


    product_dim = calculate_node_dimension_product(node_id, tree)
    if product_dim > svd_params.max_product_dim:
        greedy_opt =True
        if not is_root:
            include_parent = True
            orig_neighbors.append(parent_id)

    # Collect information about all projectors and singular values
    neighbor_projectors = {}
    neighbor_singular_values = {}
    truncated_dimensions = {}

    for neighbor_id in orig_neighbors:
        projector, singular_values = get_truncation_projector(node, 
                                                              tree.tensors[node_id], 
                                                              neighbor_id, 
                                                              svd_params)

        # Store information about all connections
        neighbor_projectors[neighbor_id] = projector
        neighbor_singular_values[neighbor_id] = singular_values
        truncated_dimensions[neighbor_id] = projector.shape[-1]

    # Optimize dimensions if needed
    if greedy_opt:
        truncated_dim_product = 1
        for dim in truncated_dimensions.values():
            truncated_dim_product *= dim

        if truncated_dim_product > svd_params.max_product_dim:
            #print(f"node: {node_id} ---> dim product: {truncated_dim_product} ")
            dimensions = tuple(truncated_dimensions.values())
            singular_values_list = tuple(neighbor_singular_values.values())
            optimized_dimensions = optimize_bond_dimensions(dimensions, 
                                                            singular_values_list, 
                                                            svd_params.max_product_dim)

            # Update projectors with optimized dimensions
            for i, neighbor_id in enumerate(orig_neighbors):
                opt_dim = optimized_dimensions[i]
                if opt_dim < truncated_dimensions[neighbor_id]:
                    # Take only the first opt_dim columns (corresponding to largest singular values)
                    neighbor_projectors[neighbor_id] = neighbor_projectors[neighbor_id][..., :opt_dim]

    # Insert projector for parent
    if not is_root and include_parent:
        insert_parent_projection_operator_and_conjugate(parent_id,
                                                        node_id,
                                                        neighbor_projectors[parent_id],
                                                        tree)

    # Insert projectors for children
    for child_id in orig_children:
        insert_child_projection_operator_and_conjugate(child_id,
                                                        node_id,
                                                        neighbor_projectors[child_id],
                                                        tree)
    # Contract all projectors with the node
    contract_projectors(tree, node_id, contract_parent = include_parent)

    # This makes the former children, the new children
    for child_id in orig_children:
        if not tree.nodes[child_id].is_leaf():
            tree.move_orthogonalization_center(child_id)
            greedy_truncate_neighbours(child_id, tree, svd_params)

def single_greedy_truncate_neighbours(node_id: str,
                                    tree: TreeTensorNetworkState,
                                    svd_params: SVDParameters):
    """
    Truncates the specified node along all neghbours.
    Args:
        node_id (str): The id of the node to truncate.
        tree (TreeTensorNetworkState): The tree tensor network to truncate.
        svd_params (SVDParameters): The parameters for the SVD truncation.
    """

    assert tree.orthogonality_center_id == node_id, \
        "The orthogonality center must be the node to truncate."
    #assert tree.is_in_canonical_form(node_id)

    node = tree.nodes[node_id]

    orig_children = copy(node.children)
    orig_neighbors = list(orig_children)
    parent_id = node.parent
    is_root = parent_id is None

    if not is_root:
        orig_neighbors.append(parent_id)

    # Collect information about all projectors and singular values
    neighbor_projectors = {}
    neighbor_singular_values = {}
    truncated_dimensions = {}

    for neighbor_id in orig_neighbors:
        projector, singular_values = get_truncation_projector(node, 
                                                              tree.tensors[node_id], 
                                                              neighbor_id, 
                                                              svd_params)

        # Store information about all connections
        neighbor_projectors[neighbor_id] = projector
        neighbor_singular_values[neighbor_id] = singular_values
        truncated_dimensions[neighbor_id] = projector.shape[-1]

    # Optimize dimensions if needed
    truncated_dim_product = 1
    for dim in truncated_dimensions.values():
        truncated_dim_product *= dim

    if truncated_dim_product > svd_params.max_product_dim:
        #print(f"node: {node_id} ---> dim product: {truncated_dim_product} ")
        dimensions = tuple(truncated_dimensions.values())
        singular_values_list = tuple(neighbor_singular_values.values())
        optimized_dimensions = optimize_bond_dimensions(dimensions, 
                                                        singular_values_list, 
                                                        svd_params.max_product_dim)

        # Update projectors with optimized dimensions
        for i, neighbor_id in enumerate(orig_neighbors):
            opt_dim = optimized_dimensions[i]
            if opt_dim < truncated_dimensions[neighbor_id]:
                # Take only the first opt_dim columns (corresponding to largest singular values)
                neighbor_projectors[neighbor_id] = neighbor_projectors[neighbor_id][..., :opt_dim]

    # Insert projector for parent
    if not is_root:
        insert_parent_projection_operator_and_conjugate(parent_id,
                                                        node_id,
                                                        neighbor_projectors[parent_id],
                                                        tree)

    # Insert projectors for children
    for child_id in orig_children:
        insert_child_projection_operator_and_conjugate(child_id, 
                                                 node_id, 
                                                 neighbor_projectors[child_id], 
                                                 tree)

    contract_projectors(tree, node_id, contract_parent = not is_root)

def sweeping_greedy_truncation(tree: TreeTensorNetworkState,
                                trunc_path: list,
                                svd_params: SVDParameters) -> TreeTensorNetworkState:
    """
    Performs a collective truncation of the tree tensor network state along the specified path.
    Args:
        tree (TreeTensorNetworkState): The tree tensor network state to truncate.
        trunc_path (list): The path along which to perform the truncation.
        svd_params (SVDParameters): Parameters for SVD truncation.
        **kwargs: Additional keyword arguments to be passed to the SVD splitting functions.
    """
    svd_params.max_bond_dim = float('inf')
    if svd_params.max_product_dim is None or svd_params.max_product_dim == float('inf'):
        warn("The greedy truncation is not effective with the current SVD parameters.")

    for node_id in trunc_path:
        if node_id == trunc_path[0]:
            if trunc_path[0] != tree.orthogonality_center_id or tree.orthogonality_center_id is None:
               tree.move_orthogonalization_center(trunc_path[0])
            single_greedy_truncate_neighbours(node_id, tree, svd_params)
        else:
            tree.move_orthogonalization_center(node_id)
            single_greedy_truncate_neighbours(node_id, tree, svd_params)