"""
Modul that contains utility functions for contractions, i.e. those needed
 in multiple kinds of contraction to avoid code duplication.
"""

from __future__ import annotations
from typing import List, Union, Tuple

from ..node import Node

def determine_index_with_ignored_leg(node: Node,
                                     neighbour_id: str,
                                     ignoring_node_id: str) -> int:
    """
    Sometimes when contracting all the neighbouring cached environments, we
     want to ignore the leg to a specific neighbour node. This means we do
     not want to contract that leg. This function determines the leg index
     of the current tensor, that should actually be contracted. This means
     earlier contractions are already taken into account.
    """
    neighbour_index = node.neighbour_index(neighbour_id)
    ignoring_index = node.neighbour_index(ignoring_node_id)
    assert ignoring_index != neighbour_index, "The next node should not be touched!"
    tensor_index_to_neighbour = int(ignoring_index < neighbour_index)
    return tensor_index_to_neighbour

def get_equivalent_legs(node1: Node,
                        node2: Node,
                        ignore_legs: Union[None,List[str]]) -> Tuple[List[int],List[int]]:
    """
    Get the equivalent legs of two nodes. This is useful when contracting
     two nodes with equal neighbour identifiers, that may potentially be in
     different orders. Some neighbours may also be ignored.
    
    Args:
        node1 (Node): The first node.
        node2 (Node): The second node.
        ignore_legs (Union[None,List[str]]): The legs to ignore.
    
    Returns:
        Tuple[List[int],List[int]]: The equivalent legs of the two nodes.
         This means the indeces of legs to the same neighbour are at the same
         position in each list.
    """
    if ignore_legs is None:
        ignore_legs = []
    legs1 = []
    legs2 = []
    for neighbour_id in node1.neighbouring_nodes():
        if neighbour_id in ignore_legs:
            continue
        legs1.append(node1.neighbour_index(neighbour_id))
        legs2.append(node2.neighbour_index(neighbour_id))
    return legs1, legs2
