"""
This module implements the recursive truncation of a tree tensor network state.

This is for example used in the BUG method.
"""

from copy import copy
from numpy import ndarray

from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.core.node import Node
from pytreenet.core.leg_specification import LegSpecification
from pytreenet.util.tensor_splitting import SVDParameters, truncated_tensor_svd

def optimize_bond_dimensions(dimensions, singular_values, max_product):
    """
    Optimizes bond dimensions to ensure their product stays below max_product.
    
    Args:
        dimensions: Tuple of current dimensions for each bond
        singular_values: Tuple of arrays, each containing singular values for a bond
        max_product: Maximum allowed product of dimensions
        
    Returns:
        Tuple of optimized dimensions
    """
    # Check if product already satisfies constraint
    current_product = 1
    for dim in dimensions:
        current_product *= dim
        
    if current_product <= max_product:
        return dimensions  

    
    # Start with minimum dimensions
    n_bonds = len(dimensions)
    optimized_dims = [1] * n_bonds
    
    # Greedy algorithm to allocate remaining dimensions
    while True:
        best_bond = -1
        best_value = -1
        
        # Try increasing each bond dimension by 1
        for i in range(n_bonds):
            current_dim = optimized_dims[i]
            
            # Skip if already at maximum possible dimension
            if current_dim >= dimensions[i] or current_dim >= len(singular_values[i]):
                continue
                
            # Get value of next singular value
            next_sv = singular_values[i][current_dim]
            
            # Check if adding this dimension keeps us under the limit
            temp_dims = optimized_dims.copy()
            temp_dims[i] += 1
            new_product = 1
            for d in temp_dims:
                new_product *= d
                
            if new_product <= max_product and next_sv > best_value:
                best_value = next_sv
                best_bond = i
        
        if best_bond == -1:
            break  # No more dimensions can be added
            
        # Add one dimension to the best bond
        optimized_dims[best_bond] += 1
        
    #print(f"Initial dimensions: {dimensions} ---> Optimized dimensions: {optimized_dims}")
    return tuple(optimized_dims)

def identity_id(child_id: str, node_id: str) -> str:
    """
    Returns the identifier for the identity tensor.

    Args:
        child_id (str): The identifier of the child node.
        node_id (str): The identifier of the node.
    
    Returns:
        str: The identifier for the identity tensor.

    """
    return f"{child_id}_identity_{node_id}"

def projector_identifier(child_id: str, node_id: str,
                         star: bool) -> str:
    """
    Returns the identifier for the projector tensor.

    Args:
        child_id (str): The identifier of the child node.
        node_id (str): The identifier of the node.
        star (bool): Whether the projector is the conjugated or not.
    
    Returns:
        str: The identifier for the projector tensor.

    """
    if star:
        return f"{child_id}_projectorstar_{node_id}"
    return f"{child_id}_projector_{node_id}"

def get_truncation_projector(node: Node,
                             node_tensor: ndarray,
                             child_id: str,
                             svd_parameters: SVDParameters) -> tuple:
    """
    Finds the projector that truncates a node for a given leg and returns singular values.

    This corresponds to finding P_i for child=i in the reference.

    Args:
        node (GraphNode): The node to truncate.
        node_tensor (ndarray): The tensor of the node.
        child_id (str): The identifier of the child for which to truncate.
        svd_parameters (SVDParameters): The parameters for the SVD.
    
    Returns:
        tuple: (projector, singular_values) The projector that truncates the node for 
               the given child and the associated singular values.
    
    """
    other_legs = list(range(node.nlegs()))
    child_index = node.neighbour_index(child_id)
    other_legs.pop(child_index)
    
    # Get both projector and singular values from SVD
    projector, s, _ = truncated_tensor_svd(node_tensor,
                                        (child_index, ),
                                        tuple(other_legs),
                                        svd_parameters)
    return projector, s

def insert_child_projection_operator_and_conjugate(child_id: str,
                                            node_id: str,
                                            projector: ndarray,
                                            tree: TreeTensorNetwork
                                            ):
    """
    Inserts the projector and its conjugate between two nodes.

    Args:
        child_id (str): The id of the child node.
        node_id (str): The id of the parent node.
        projector (ndarray): The projector to insert.

    """
    id_identity = identity_id(child_id, node_id)
    tree.insert_identity(child_id, node_id,
                                new_identifier=id_identity)
    proj_star_legs = LegSpecification(node_id,[],[])
    proj_legs = LegSpecification(None,[child_id],[])
    tree.split_node_replace(id_identity,
                                    projector.conj(),
                                    projector.T,
                                    projector_identifier(node_id, child_id, True),
                                    projector_identifier(node_id, child_id, False),
                                    proj_star_legs,
                                    proj_legs
                                    )

def insert_parent_projection_operator_and_conjugate(parent_id: str,
                                      node_id: str,
                                      projector: ndarray,
                                      tree: TreeTensorNetwork):
    """
    Inserts the projector and its conjugate between a node and its parent.

    Args:
        node_id (str): The id of the node.
        parent_id (str): The id of the parent node.
        projector (ndarray): The projector to insert.

    """
    id_identity = identity_id(node_id, parent_id)
    tree.insert_identity(node_id, parent_id,
                                new_identifier=id_identity)
    proj_star_legs = LegSpecification(None,[node_id],[])
    proj_legs = LegSpecification(parent_id,None,[])
    tree.split_node_replace(id_identity,
                                    projector.conj(),
                                    projector.T,
                                    projector_identifier(node_id ,parent_id, True),
                                    projector_identifier(node_id ,parent_id, False),
                                    proj_star_legs,
                                    proj_legs
                                    )
    
def truncate_node(node_id: str,
                  tree: TreeTensorNetwork,
                  svd_params: SVDParameters):
    """
    Truncates the node with the given id.

    Args:
        node_id (str): The id of the node to truncate.

    """
    node = tree.nodes[node_id]
    orig_children = copy(node.children)
    for child_id in orig_children:
        projector , _ = get_truncation_projector(node,
                                            tree.tensors[node_id],
                                            child_id,
                                            svd_params)
        insert_child_projection_operator_and_conjugate(child_id,
                                                    node_id,
                                                    projector,
                                                    tree)
    tree.contract_all_children(node_id)
    # Now we contract all the projectors into the former children
    for child_id in copy(tree.nodes[node_id].children):
        child_node = tree.nodes[child_id]
        assert len(child_node.children) == 1, "Projector node has more than one child!"
        orig_child_id = child_node.children[0]
        tree.contract_all_children(child_id,
                                        new_identifier=orig_child_id)
    # This makes the former children, the new children
    for child_id in orig_children:
        truncate_node(child_id, tree, svd_params)

def truncate_single_node(node_id: str,
                        tree: TreeTensorNetwork,
                        svd_params: SVDParameters):
    """
    Truncates only the specified node (not recursively down to leaves).
    
    Unlike truncate_node, this function processes both child and parent connections,
    but doesn't recursively process child nodes.
    
    Args:
        node_id (str): The id of the node to truncate.
        tree (TreeTensorNetwork): The tree tensor network to truncate.
        svd_params (SVDParameters): The parameters for the SVD truncation.
    """
    node = tree.nodes[node_id]
    
    orig_children = copy(node.children)
    orig_neighbors = list(orig_children) 
    parent_id = node.parent
    is_root = parent_id is None
    if not is_root:
        orig_neighbors.append(parent_id)
    
    # Collect information about all connections
    neighbor_projectors = {}
    neighbor_singular_values = {}
    initial_dimensions = {}
    
    for neighbor_id in orig_neighbors:

        projector, singular_values = get_truncation_projector(node, 
                                                              tree.tensors[node_id], 
                                                              neighbor_id, 
                                                              svd_params)
        
        # Store information about all connections
        neighbor_projectors[neighbor_id] = projector
        neighbor_singular_values[neighbor_id] = singular_values
        initial_dimensions[neighbor_id] = projector.shape[-1]
    
    # Optimize dimensions if needed
    dimensions = tuple(initial_dimensions.values())
    singular_values_list = tuple(neighbor_singular_values.values())
    optimized_dimensions = optimize_bond_dimensions(dimensions, 
                                                    singular_values_list, 
                                                    svd_params.max_effective_ham_dim)
    
    # Update projectors with optimized dimensions
    for i, neighbor_id in enumerate(orig_neighbors):
        opt_dim = optimized_dimensions[i]
        if opt_dim < initial_dimensions[neighbor_id]:
            # Take only the first opt_dim columns (corresponding to largest singular values)
            neighbor_projectors[neighbor_id] = neighbor_projectors[neighbor_id][..., :opt_dim]
    
    # Insert projectors for children
    for child_id in orig_children:
        insert_child_projection_operator_and_conjugate(child_id, 
                                                 node_id, 
                                                 neighbor_projectors[child_id], 
                                                 tree)
    
    # Insert projector for parent
    if not is_root:
        insert_parent_projection_operator_and_conjugate(parent_id, 
                                          node_id, 
                                          neighbor_projectors[parent_id], 
                                          tree)
    
    # Contract children projectors
    tree.contract_all_children(node_id)
    
    # For non-root nodes, contract with parent's projector
    if not is_root:
        parent_projector_star_id = projector_identifier(node_id, parent_id, True)
        if parent_projector_star_id in tree.nodes:
            # Contract the projector star to the node
            tree.contract_nodes(node_id, parent_projector_star_id, new_identifier=node_id)
            # Contract the other projector with the parent node
            parent_projector_id = projector_identifier(node_id, parent_id, False)
            if parent_projector_id in tree.nodes:
                tree.contract_nodes(parent_id, parent_projector_id, new_identifier=parent_id)
    
    # Contract all projectors with children 
    for child_id in copy(tree.nodes[node_id].children):
        child_node = tree.nodes[child_id]
        if len(child_node.children) == 1: 
            orig_child_id = child_node.children[0]
            tree.contract_all_children(child_id, new_identifier=orig_child_id)

def calculate_node_dimension_product(node_id: str, tree: TreeTensorNetwork) -> int:
    """
    Calculates the product of all dimensions connected to a node.
    
    Args:
        node_id (str): The id of the node to check.
        tree (TreeTensorNetwork): The tree tensor network.
        
    Returns:
        int: The product of all dimensions (neighbors + open legs).
    """
    node = tree.nodes[node_id]
    neighbors = node.neighbouring_nodes()
    
    total_product = 1
    for neighbor_id in neighbors:
        leg_idx = node.neighbour_index(neighbor_id)
        total_product *= node.shape[leg_idx]
    
    #for open_leg_idx in node.open_legs:
    #    total_product *= node.shape[open_leg_idx]
    
    return total_product

def post_truncate_node(node_id: str, tree: TreeTensorNetwork, svd_params: SVDParameters):
    """    
    Args:
        node_id (str): The id of the node to start traversal from.
        tree (TreeTensorNetwork): The tree tensor network.
        svd_params (SVDParameters): The parameters for the SVD truncation.
    """
    dim_product = calculate_node_dimension_product(node_id, tree)

    if dim_product > svd_params.max_effective_ham_dim: 
        tree.move_orthogonalization_center(node_id)
        #print(f"Node {node_id} has dimension product {dim_product} > {svd_params.max_effective_ham_dim}")
        truncate_single_node(node_id, tree, svd_params) 
    
    children = copy(tree.nodes[node_id].children)
    for child_id in children:
        post_truncate_node(child_id, tree, svd_params)

def recursive_truncation(tree: TreeTensorNetwork,
                         svd_params: SVDParameters) -> TreeTensorNetwork:
    """
    Truncates the tree tensor network using the recursive truncation method.

    Args:
        tree (TreeTensorNetwork): The tree tensor network to be truncated.
        svd_params (SVDParameters): The parameters for the SVD truncation.

    Returns:
        TreeTensorNetwork: The modified truncated tree tensor network.
    """
    root_id = tree.root_id
    if root_id != tree.orthogonality_center_id or tree.orthogonality_center_id is None:
        tree.canonical_form(root_id)
    
    # 1: Standard hierarchical truncation
    truncate_node(root_id, tree, svd_params)
    
    # 2: Post-truncation with max_effective_ham_dim
    post_svd_params = copy(svd_params)
    post_svd_params.sum_trunc = False
    post_svd_params.rel_tol = float('-inf')
    post_svd_params.total_tol = float('-inf')

    post_truncate_node(root_id, tree, post_svd_params)
    
    # Restore orthogonality center to root
    if tree.orthogonality_center_id != root_id:
        tree.move_orthogonalization_center(root_id)
    
    return tree