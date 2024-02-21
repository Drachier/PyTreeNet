from __future__ import annotations

from copy import copy

import numpy as np

from .util import copy_object

def completely_contract_tree(ttn: TreeTensorNetwork, to_copy: bool=False) -> TreeTensorNetwork:
    """
    Completely contracts the given tree_tensor_network by combining all nodes.
    (WARNING: Can get very costly very fast. Only use for debugging.)

    Args:
        ttn (TreeTensorNetwork): The TTN to be contracted.
        to_copy (bool): Wether or not the contraction should be perfomed on a deep copy.
            Default is False.
    """
    work_ttn = copy_object(ttn, deep=to_copy)

    root_id = work_ttn.root_id
    _completely_contract_tree_rec(work_ttn, root_id)

    return work_ttn

def _completely_contract_tree_rec(work_ttn: TreeTensorNetwork, current_node_id: str):
    """
    Recursively contracts the complete subtree of the given node.

    Args:
        work_ttn (TreeTensorNetwork): The TTN to be contracted
        current_node_id (str): The node into which we want to contract the subtree.
    """
    # print(current_node_id)
    current_node = work_ttn.nodes[current_node_id]
    children = copy(current_node.children)
    for child_id in children:
        # Contracts the complete subtree into this child
        _completely_contract_tree_rec(work_ttn, child_id)
        work_ttn.contract_nodes(current_node_id, child_id, new_identifier=current_node_id)

def _contract_same_structure_nodes(node_id: str, ttn1: TreeTensorNetwork,
                                   ttn2: TreeTensorNetwork) -> np.ndarray:
    """
    Contracts the two nodes in the tensor networks that correspond to the same identifier.

    Args:
        node_id (str): The identifier giving the position in the TTNs.
        ttn1 (TreeTensorNetwork): One TTN to be contracted
        ttn2 (TreeTensorNetwork): Second TTN to be contracted

    Returns:
        (np.ndarray) : The contraction result
    """
    node1 = ttn1.nodes[node_id]
    if node1.is_leaf():
        open_legs = node1.open_legs
        resulting_tensor = np.tensordot(ttn1.tensors[node_id], ttn2.tensors[node_id],
                                        axes=(open_legs, open_legs))
        return resulting_tensor

    children = node1.children
    result_tensors = {}
    for child_id in children:
        # This tensor will have exactly two legs.
        # Leg 0 is contracted with node1 and leg 1 with node2.
        child_tensor = _contract_same_structure_nodes(child_id, ttn1, ttn2)
        result_tensors[child_id] = child_tensor

    open_legs = node1.open_legs
    for child_id in children:
        result_tensor = result_tensors[child_id]
        ttn1.absorb_tensor_into_neighbour_leg(node_id, child_id, result_tensor, 0)
    # Now the tensor of node1 contains all the children results
    n_legs = node1.nlegs()
    if node1.is_root():
        contracting_legs = list(range(n_legs))
    else:
        contracting_legs = list(range(1, n_legs))
    resulting_tensor = np.tensordot(ttn1.tensors[node_id], ttn2.tensors[node_id],
                                    axes=(contracting_legs, contracting_legs))
    return resulting_tensor

def contract_two_ttn(ttn1: TreeTensorNetwork, ttn2: TreeTensorNetwork) -> complex:
    """
    Contracts two TTN with the same structure.
    Assumes both TTN use the same identifiers for the nodes.

    Args:
        ttn1 (TreeTensorNetwork): One TTN to be contracted
        ttn2 (TreeTensorNetwork): Second TTN to be contracted

    Returns:
        complex: The contraction result.
    """
    return complex(_contract_same_structure_nodes(ttn1.root_id, ttn1, ttn2))
