"""
Module to find the update path of a TDVP algorithm.
"""
from __future__ import annotations
from typing import List
from ...core.tree_structure import TreeStructure
from enum import Enum

class PathFinderMode(Enum):
      LeafToLeaf = "LeafToLeaf"
      LeafToRoot = "LeafToRoot"

class TDVPUpdatePathFinder():
    """
    Base class to construct the update path for a TDVP algorithm.

    Attributes:
        state (TreeStructure): The tree structure used for path finding.
        mode (PathFinderMode): The update path strategy mode.
                               Can be either `LeafToLeaf` or `LeafToRoot`.
    """
    def __init__(self, state: TreeStructure,
                 mode: PathFinderMode = PathFinderMode.LeafToLeaf) -> None:
        self.state = state
        self.mode = mode

        if self.mode == PathFinderMode.LeafToRoot:
            self._finder = TDVPUpdatePathFinder_LeafToRoot(self.state)
        elif self.mode == PathFinderMode.LeafToLeaf:
            self._finder = TDVPUpdatePathFinder_LeafToLeaf(self.state)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def find_path(self) -> List[str]:
        """
        Finds the complete update path using the selected path finding strategy.

        Returns:
            List[str]: The ordered list of node identifiers constituting the update path.
        """
        return self._finder.find_path()

class TDVPUpdatePathFinder_LeafToRoot():
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
        self.state = state
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

class TDVPUpdatePathFinder_LeafToLeaf():
    """
    Constructs a leaf-to-leaf update path for a TDVP algorithm:

      1) Identifies two leaves L_A, L_B that are farthest apart.
      2) main_path = path_from_to(L_A, L_B).
      3) Visits *all* off-path subtrees (including possibly the root, if it's not
         on main_path), so that every node is visited exactly once.

    Attributes:
        state (TreeStructure): A copy of the original tree, used for path finding.
        start (str): One diameter leaf (L_A).
        end (str):   The other diameter leaf (L_B).
        main_path (List[str]): The direct path from L_A to L_B.
    """

    def __init__(self, state) -> None:
        self.state = state
        self.start, self.end = self._find_two_diameter_leaves()
        self.main_path = self.state.path_from_to(self.start, self.end)

    def _find_two_diameter_leaves(self) -> Tuple[str, str]:
        """
        Finds two leaves L_A, L_B that maximize distance by explicitly
        checking all pairs of leaves.

        Returns:
            (L_A, L_B): Identifiers of the diameter leaves.
        """
        leaves = self.state.get_leaves(include_root = True)

        best_dist = -1
        best_pair = (leaves[0], leaves[0])

        for i in range(len(leaves)):
            dist_i = self.state.distance_to_node(leaves[i])
            for j in range(i + 1, len(leaves)):
                d = dist_i[leaves[j]]
                if d > best_dist:
                    best_dist = d
                    best_pair = (leaves[i], leaves[j])

        return best_pair

    def find_path(self) -> List[str]:
        """
        Returns the full traversal order (covering all nodes in the tree),
        starting at L_A, ending at L_B, and visiting any branch subtrees that
        are not on the main path.
        - For each 'branch_origin' in self.main_path (in order from start to end):
              1) Include its parent subtree if off-path
              2) Include child subtrees if off-path
              3) Finally include branch_origin itself
        """
        visited = set()
        full_path = []

        for branch_origin in self.main_path:
            parent_subtree = self._visit_offpath_subtree_parents(branch_origin, visited)
            children_subtree = self._visit_offpath_subtree_children(branch_origin, visited)

            full_path.extend(parent_subtree)
            full_path.extend(children_subtree)

            if branch_origin not in visited:
                visited.add(branch_origin)
                full_path.append(branch_origin)

        return full_path

    def _visit_offpath_subtree_parents(self, branch_origin: str, visited: set) -> List[str]:
        """
        If branch_origin has a parent that is NOT on main_path and not visited,
        we gather that parent's entire subtree (potentially climbing further up
        if that parent also has an off-path parent, etc.).

        In a standard tree, a node has exactly one parent, so we stop climbing
        once we incorporate that parent subtree.
        """
        node = self.state.nodes[branch_origin]
        path_collected = []

        while (not node.is_root()
               and node.parent not in self.main_path
               and node.parent not in visited):
            p_id = node.parent
            # gather the parent's subtree (children first, then the node)
            path_collected.extend(self._subtree_path_rec(p_id, visited))
            # After handling that parent once, break, because we've already included climbing further up if needed.
            break

        return path_collected

    def _visit_offpath_subtree_children(self, branch_origin: str, visited: set) -> List[str]:
        """
        For the main-path node `branch_origin`, gather all child subtrees that
        are not on the main path.
        """
        node = self.state.nodes[branch_origin]
        offpath_kids = [c for c in node.children if c not in self.main_path]
        sub_path = []

        for child_id in offpath_kids:
            sub_path.extend(self._subtree_path_rec(child_id, visited))

        return sub_path

    def _subtree_path_rec(self, node_id: str, visited: set) -> List[str]:
        """
        Recursively collects all nodes in the subtree rooted at node_id,
        in a children-first fashion: first incorporate child subtrees, 
        then the node itself.

        We skip nodes that are already visited or on the main_path to 
        prevent revisits or collisions. If this node has a parent off the 
        main path and unvisited, we also gather that as part of the upward chain.

        Args:
            node_id (str): The id of the current subtree root.
            visited (set): A set of nodes already included in the path.

        Returns:
            path (List[str]): The list of node_ids in the gathered subtree.
        """
        if node_id in visited or node_id in self.main_path:
            return []

        visited.add(node_id)
        node = self.state.nodes[node_id]
        path = []

        if (not node.is_root()
                and node.parent not in self.main_path
                and node.parent not in visited):
            path.extend(self._subtree_path_rec(node.parent, visited))

        for c_id in node.children:
            if c_id not in self.main_path and c_id not in visited:
                path.extend(self._subtree_path_rec(c_id, visited))

        path.append(node_id)

        return path

def find_orthogonalisation_path(update_path: List[str],
                                state: TreeStructure,
                                include_start: bool = False
                                ) -> List[List[str]]:
    """
    Finds the orthogonalization path for a given update path in the state.

    Args:
        update_path (List[str]): The update path to find the orthogonalization for.
        state (TreeStructure): The current state of the tree.
        include_start (bool): Whether to include the starting node in the
            orthogonalization path. Default is False.

    Returns:
        List[List[str]]: A list of sub-paths representing the orthogonalization path.
    """
    orthogonalization_path = []
    for i in range(len(update_path)-1):
        sub_path = state.path_from_to(update_path[i], update_path[i+1])
        orthogonalization_path.append(sub_path[1:] if not include_start else sub_path)
    return orthogonalization_path
