from __future__ import annotations
from typing import List, Tuple

from copy import copy

from ..util import copy_object

def completely_contract_tree(ttn: TreeTensorNetwork,
                             to_copy: bool=False) -> Tuple[np.ndarray, List[str]]:
    """
    Completely contracts the given tree_tensor_network by combining all nodes.
    (WARNING: Can get very costly very fast. Only use for debugging.)

    Args:
        ttn (TreeTensorNetwork): The TTN to be contracted.
        to_copy (bool): Wether or not the contraction should be perfomed on a deep copy.
            Default is False.

    Returns:
        Tuple[np.ndarray, List[str]]: The contracted TTN and the list of the
            identifiers of the contracted nodes in the order they were contracted.
            The latter is very useful for debugging.
    """
    work_ttn = copy_object(ttn, deep=to_copy)
    contraction_oder = []
    root_id = work_ttn.root_id
    _completely_contract_tree_rec(work_ttn, root_id, contraction_oder)
    return work_ttn.tensors[work_ttn.root_id], contraction_oder

def _completely_contract_tree_rec(work_ttn: TreeTensorNetwork,
                                  current_node_id: str,
                                  contraction_order: List[str]):
    """
    Recursively contracts the complete subtree of the given node.

    Args:
        work_ttn (TreeTensorNetwork): The TTN to be contracted
        current_node_id (str): The node into which we want to contract the subtree.
    """
    current_node = work_ttn.nodes[current_node_id]
    children = copy(current_node.children)
    contraction_order.append(current_node_id)
    for child_id in children:
        # Contracts the complete subtree into this child
        _completely_contract_tree_rec(work_ttn, child_id,
                                      contraction_order)
        work_ttn.contract_nodes(current_node_id, child_id,
                                new_identifier=current_node_id)
