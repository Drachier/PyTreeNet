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
                             svd_parameters: SVDParameters) -> ndarray:
    """
    Finds the projector that truncates a node for a given leg.

    This corresponds to finding P_i for child=i in the reference.

    Args:
        node (GraphNode): The node to truncate.
        node_tensor (ndarray): The tensor of the node.
        child_id (str): The identifier of the child for which to truncate.
        svd_parameters (SVDParameters): The parameters for the SVD.
    
    Returns:
        ndarray: The projector that truncates the node for the given child.
            Has the leg order (child_leg,new_leg).
    
    """
    other_legs = list(range(node.nlegs()))
    child_index = node.neighbour_index(child_id)
    other_legs.pop(child_index)
    projector, _, _ = truncated_tensor_svd(node_tensor,
                                        (child_index, ),
                                        tuple(other_legs),
                                        svd_parameters)
    return projector

def insert_projection_operator_and_conjugate(child_id: str,
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
        projector = get_truncation_projector(node,
                                            tree.tensors[node_id],
                                            child_id,
                                            svd_params)
        insert_projection_operator_and_conjugate(child_id,
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

def recursive_truncation(tree: TreeTensorNetwork,
                         svd_params: SVDParameters
                         ) -> TreeTensorNetwork:
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
    truncate_node(root_id, tree, svd_params)
    return tree
