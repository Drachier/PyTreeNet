"""
Module to find the update path of a TDVP algorithm.
"""
from __future__ import annotations
from typing import List
from copy import deepcopy
from ...core.tree_structure import TreeStructure

class TDVPUpdatePathFinder():
    """
    Constructs the update path of a TDVP algorithm.

    The update path should minimise the number of orthogonalisations, i.e.
    QR-decompositions, during the time-evolution. To this end the start and end
    node are chosen to be the two leafs which are furthest away from each
    other.

    Attributes:
        state (TreeStructure): The tree topology to find the update path on.
        start (GraphNode): The node to start the update path at.
        main_path (List[str]): The main path, i.e. the longest path in the
            tree along which to run. For an MPS this would be the only path.
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

    def find_furthest_non_visited_leaf(self, path: List[str]) -> str:
        """
        Finds the leaf that is furthest from the origin once and was not yet
        visited.

        Args:
            path (List[sr]): A list of all node_ids already visited.

        Returns:
            str: The identifier of the leaf which is furthest away from the
                main path and has not been visited yet.
        """
        all_leaves = self.state.get_leaves()
        non_visited_leaves = [leaf for leaf in all_leaves
                              if leaf not in path]
        distances_from_root = self.state.distance_to_node(self.state.root_id)
        leaf_distances = {leaf_id: distance
                          for leaf_id, distance in distances_from_root.items()
                          if leaf_id in non_visited_leaves}
        assert len(leaf_distances) != 0
        return max(leaf_distances, key=leaf_distances.get)

    def find_main_path_down_from_root(self, path: List[str]) -> List[str]:
        """
        Finds the main path which to traverse from the root to the last
        leaf.

        Args:
            path (List[str]): The path already traversed.

        Returns:
            List[str]: Main path from the root to the last leaf.
                `[root, node, node, ... , leaf]`
        """
        final_node_id = self.find_furthest_non_visited_leaf(path)
        main_path_down = self.state.find_path_to_root(final_node_id)
        main_path_down.reverse()
        return main_path_down

    def _branch_downwards_origin_is_root(self,
                                         main_path_down: List[str]) -> List[str]:
        """
        Finds the path going through the branches starting at the origin.
        
        This specifically excludes the branch already traversed and the
        branch used to go back down the tree.

        Args:
            main_path_down (List[str]): The main path down the tree.

        Returns:
            List[str]: The path through the branch ending with the root.
        """
        children_ids = [child_id
                        for child_id in self.state.nodes[self.state.root_id].children
                        if child_id not in (self.main_path[-2],main_path_down[1])]
        branch_path = []
        for child_id in children_ids:
            branch_path.extend(self._path_for_branch_rec(child_id))
        branch_path.append(self.state.root_id)
        return branch_path

    def _branch_path_downwards(self, branch_origin: str,
                               main_path_down: List[str]) -> List[str]:
        """
        Finds the path through a branch while going down from the origin.

        Args:
            branch_origin (str): current node identifier
            main_path_down (List[str]): The main path down the tree.

        Returns:
            List[str]: The path through the branch ending with the
                branch_origin.
        """
        children_ids = [child_id
                        for child_id in self.state.nodes[branch_origin].children
                        if child_id not in main_path_down]
        branch_path = []
        for child_id in children_ids:
            branch_path.extend(self._path_for_branch_rec(child_id))
        branch_path.append(branch_origin)
        return branch_path

    def path_down_from_root(self, path: List[str]) -> List[str]:
        """
        Finds the complete path from the root to the last leaf.

        Args:
            path (List[str]): The path that was already traversed.

        Returns:
            List[str]: The complete path from the root to the last leaf.
                `[root, node, node, ... , leaf]`
        """
        root_id = self.state.root_id
        assert root_id is not None
        if self.state.nodes[root_id].has_x_children(1):
            return [root_id]
        main_path_down = self.find_main_path_down_from_root(path)
        down_path = []
        for branch_origin in main_path_down:
            if branch_origin == self.state.root_id:
                branch_path = self._branch_downwards_origin_is_root(main_path_down)
            else:
                branch_path = self._branch_path_downwards(branch_origin, main_path_down)
            down_path.extend(branch_path)
        return down_path

    def find_path(self) -> List[str]:
        """
        Finds the complete update path for a TDVP along a main path.
        
        All nodes in branches are added before the branch origin in the main
        path.
        """
        path = []
        for branch_origin in self.main_path:
            if branch_origin != self.state.root_id:
                path.extend(self.path_for_branch(branch_origin))
            else:
                path.extend(self.path_down_from_root(path))
        return path
