"""
Modul that contains utility functions for contractions, i.e. those needed
 in multiple kinds of contraction to avoid code duplication.
"""

from __future__ import annotations

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
