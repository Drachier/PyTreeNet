from __future__ import annotations
from typing import List
from copy import deepcopy
from ...tree_structure import TreeStructure

class TDVPUpdatePathFinder():
    """
    Constructs the update path for a tdvp algorithm that minimises the number
     of orthogonalisations (i.e. QR-decomps) required during the time
     evolution.
    """

    def __init__(self, state: TreeStructure) -> None:
        self.state = deepcopy(state)
        self.start = self.find_start_node_id()
        self.main_path = self.state.find_path_to_root(self.start)

    def find_start_node_id(self) -> str:
        """
        Finds the node id at which to start the TDVP update.
        This would be the initial orthogonalisation center of the state that
         is to be time-evolved. Currently, we assume this site is the leaf
         furthest away from the root.
        # TODO: Allow option to find the leaves furthest away from one another.

        Returns:
            str: The node_id at which to start the update.
        """
        distances_from_root = self.state.distance_to_node(self.state.root_id)
        return max(distances_from_root,
                   key=distances_from_root.get)

    def path_for_branch(self, branch_origin: str) -> List[str]:
        """
        Finds the node_ids that need to be visited after the last main path
         node and before branch_origin.

        Args:
            branch_origin (str): The identifier of the node in the main path
             which has the branch as a subtree.

        Returns:
            List[str]: The desired path. The children appear before their
             parent.
        """
        node = self.state.nodes[branch_origin]
        children_ids = [child_id for child_id in node.children
                        if child_id not in self.main_path]
        branch_path = []
        for child_id in children_ids:
            branch_path.extend(self._path_for_branch_rec(child_id))
        branch_path.append(branch_origin)
        return branch_path

    def _path_for_branch_rec(self, node_id: str) -> List[str]:
        node = self.state.nodes[node_id]
        if node.is_leaf():
            return [node_id]
        path = []
        for child_id in node.children:
            path.extend(self._path_for_branch_rec(child_id))
        path.append(node_id)
        return path

    def find_path(self) -> List[str]:
        """
        Finds the complete update path for a TDVP along a main path.
         All nodes in branches are added before the branch origin in the main
         path.
        """
        path = []
        for branch_origin in self.main_path:
            path.append(self.path_for_branch(branch_origin))
        return path

