"""
This module provides utility functions for SVD and Greedy truncation operations.
"""
from uuid import uuid1
from copy import copy
from numpy import ndarray

from ...time_evolution.time_evo_util.update_path import SweepingUpdatePathFinder, PathFinderMode
from ...ttns.ttns import TreeTensorNetworkState
from ...core.node import Node
from ...core.leg_specification import LegSpecification
from ...util.tensor_splitting import SVDParameters, truncated_tensor_svd
from ..ttn import TreeTensorNetwork


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
                             neighbour_id: str,
                             svd_parameters: SVDParameters) -> ndarray:
    """
    Finds the projector that truncates a node for a given leg.

    This corresponds to finding P_i for child=i in the reference.

    Args:
        node (GraphNode): The node to truncate.
        node_tensor (ndarray): The tensor of the node.
        neighbour_id (str): The identifier of the child for which to truncate.
        svd_parameters (SVDParameters): The parameters for the SVD.

    Returns:
        ndarray: The projector that truncates the node for the given child.
            Has the leg order (child_leg,new_leg).

    """
    other_legs = list(range(node.nlegs()))
    child_index = node.neighbour_index(neighbour_id)
    other_legs.pop(child_index)
    projector, s, _ = truncated_tensor_svd(node_tensor,
                                        (child_index, ),
                                        tuple(other_legs),
                                        svd_parameters)
    return projector, s

def insert_parent_projection_operator_and_conjugate(parent_id: str,
                                      node_id: str,
                                      projector: ndarray,
                                      tree: TreeTensorNetworkState):
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

def insert_child_projection_operator_and_conjugate(child_id: str,
                                            node_id: str,
                                            projector: ndarray,
                                            tree: TreeTensorNetworkState
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

def calculate_node_dimension_product(node_id: str, tree: TreeTensorNetworkState) -> int:
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

def contract_projectors(tree: TreeTensorNetworkState,
                        node_id: str,
                        contract_parent: bool):
    """
    Contracts all projectors along to the tensor at node_id.
    Args:
        tree (TreeTensorNetworkState): The tree tensor network state.
        node_id (str): The identifier of the node to contract projectors for.
        contract_parent (bool): If True, the partent projector contracted.
    """
    
    # Contract children projectors
    tree.contract_all_children(node_id)

    # For non-root nodes, contract with parent's projector
    if contract_parent:
        parent_projector_star_id = tree.nodes[node_id].parent
        if parent_projector_star_id in tree.nodes:
            # Contract the projector star to the node
            tree.contract_nodes(node_id, parent_projector_star_id, new_identifier=node_id)
            # Contract the other projector with the parent node
            parent_projector_id = tree.nodes[node_id].parent
            parent_id = tree.nodes[parent_projector_id].parent
            if parent_projector_id in tree.nodes:
                tree.contract_nodes(parent_id, parent_projector_id, new_identifier=parent_id)

    # Contract all projectors with children 
    for child_id in copy(tree.nodes[node_id].children):
        child_node = tree.nodes[child_id]
        assert len(child_node.children) == 1
        orig_child_id = child_node.children[0]
        tree.contract_all_children(child_id, new_identifier=orig_child_id)

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

def move_orth_for_path(ttn, path):
    """
    Moves the orthogonalisation center and updates all required caches
    along a given path.

    Args:
        path (List[str]): The path to move from. Should start with the
            orth center and end at with the final node. If the path is empty
            or only the orth center nothing happens.
    """
    if len(path) == 0:
        return
    assert ttn.orthogonality_center_id == path[0]
    for node_id in path[1:]:
        ttn.move_orthogonalization_center(node_id)

def find_orthogonalization_path(state, trunc_path):
    """
    Finds the orthogonalization path for a given truncation path on a TTN structure.
    Args:
        state (TreeTensorNetworkState): The TTN structure to find the path on.
        trunc_path (list): A list of node identifiers representing the truncation path.
    Returns:
        list: A list of orthogonalization paths, where each path is a list of node identifiers.
    """
    orth_path = []
    for i in range(len(trunc_path)-1):
        sub_path = state.path_from_to(trunc_path[i], trunc_path[i+1])
        orth_path.append(sub_path[1::])
    return orth_path

def find_greedy_trunc_path(ttn: TreeTensorNetworkState,
                              forward: bool = True) -> list:
    """
    Finds the truncation path for a greedy truncation
    in a TTN structure. see `SweepingUpdatePathFinder` for more details.
    Args:
        ttn (TreeTensorNetworkState): The TTN structure to find the path on.
        forward (bool): forward direction if True, backward otherwise.
    Returns:
        list: A list of node identifiers representing the truncation path.
    """
    if forward:
        forward_trunc_path = []
        forward_path = SweepingUpdatePathFinder(ttn,PathFinderMode.LeafToLeaf_Forward).find_path()
        for node_id in forward_path:
            if not ttn.nodes[node_id].is_leaf():
                forward_trunc_path.append(node_id)
        return forward_trunc_path
    else:
        backward_trunc_path = []
        backward_path = SweepingUpdatePathFinder(ttn,PathFinderMode.LeafToLeaf_Backward).find_path()
        for node_id in backward_path:
            if not ttn.nodes[node_id].is_leaf():
                backward_trunc_path.append(node_id)
        return backward_trunc_path

