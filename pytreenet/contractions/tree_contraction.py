from __future__ import annotations

from copy import copy

from ..util import copy_object

def completely_contract_tree(ttn: TreeTensorNetwork,
                             to_copy: bool=False) -> TreeTensorNetwork:
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

def _completely_contract_tree_rec(work_ttn: TreeTensorNetwork,
                                  current_node_id: str):
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
