from __future__ import annotations

from typing import List, Tuple, Dict

from .node import Node


class TreeStructure():
    """
    An abstract tree tensor network (TreeStructure) tree, where each node represents a tensor,
    that is part of the network. Here a tree tensor network is a dictionary
    _nodes of tensor nodes with their identifiers as keys. No data is stored in this structure.

    General structure and parts of the codes are from treelib.tree

    Attributes
    -------
    _nodes: dict[str, TensorNode] mapping node ids (str) to TensorNode objects
    _root_id: str identifier for root node of TreeStructure
    """

    def __init__(self):
        """
        Initiates a new TreeTensorNetwork or a deep or shallow copy of a
        different one.
        """
        self._nodes = {}
        self._root_id = None

    @property
    def nodes(self):
        """
        A dict[str, Node] mapping the tensor tree node identifiers to the respective Node objects.
        """
        return self._nodes

    @property
    def root_id(self):
        """
        The root's identifier.
        """
        return self._root_id

    @root_id.setter
    def root_id(self, new_root_id: str):
        """
        Sets a new root_id
        """
        self._root_id = str(new_root_id)

    def __contains__(self, identifier: str):
        """
        Determines if a node with identifier is in the TreeStructure.
        """
        return identifier in self._nodes

    def __getitem__(self, key: str):
        """
        Return Node object associated with `key`
        """
        return self._nodes[key]

    def __len__(self):
        return len(self._nodes)

    def ensure_uniqueness(self, node_id: str):
        """
        Ensures that the given identifier is not already in use.

        Args:
            node_id (str): Identifer to check.

        Raises:
            ValueError: Raised if the a node in this tree already has this identifier.
        """
        if node_id in self._nodes:
            err_str = f"Tree already contains a node with identifier {node_id}!"
            raise ValueError(err_str)

    def ensure_existence(self, node_id: str):
        """
        Ensures that an identifier is already in this tree.

        Args:
            node_id (str): Identifier to check.

        Raises:
            ValueError: Raised if there is no node with the identifier in this tree.
        """
        if node_id not in self._nodes:
            err_str = f"Node with identifier {node_id} is not in this tree!"
            raise ValueError(err_str)

    def _add_node(self, new_node: Node):
        """
        Adds a node to the dictionary with checks.

        Only for internal use. For external additions use the functions below.
        """
        new_node_id = new_node.identifier
        self.ensure_uniqueness(new_node_id)
        self.nodes[new_node_id] = new_node

    def add_root(self, node: Node):
        """
        Adds a root Node to this tree
        """
        assert self.root_id is None, "A tree may only have one root."
        self._root_id = node.identifier
        self._nodes[node.identifier] = node

    def add_child_to_parent(self, child: Node, parent_id: str):
        """
        Adds a Node as a child to the specified parent_node.

        Args:
            child (Node): The node to be added
            parent_id (str): The identifier of the node which is to be the new parent.
        """
        self.ensure_existence(parent_id)
        self._add_node(child)
        child.add_parent(parent_id)

        child_id = child.identifier
        parent = self._nodes[parent_id]
        parent.add_child(child_id)

    def add_parent_to_root(self, new_root: Node):
        """
        Adds a parent to the root of this tree, making it the new root.
        """
        self._add_node(new_root)
        new_root.add_child(self._root_id)

        current_root = self._nodes[self._root_id]
        new_id = new_root.identifier
        current_root.add_parent(new_id)

        self._root_id = new_id

    def nearest_neighbours(self) -> List[Tuple[str, str]]:
        """
        Finds all nearest neighbour pairs in this tree.

        They are found by collecting all parent child pairs in tuples.
        The first entry is the parent identifier and the second the child identifier.

        Returns:
            nn_list (List[Tuple[str,str]]) : A list containing the identifiers of all
                nearest neighbour pairs.
        """
        nn_list = []
        for node_id in self._nodes:
            current_node = self.nodes[node_id]
            for child_id in current_node.children:
                nn_list.append((node_id, child_id))
        return nn_list

    def get_leaves(self) -> List[str]:
        """
        Returns a list with the identifiers of all leaves.
        """
        return [node_id for node_id, node in self._nodes.items()
                if node.is_leaf()]

    def distance_to_node(self, center_node_id: str) -> Dict[str, int]:
        """
        Finds the distance of every node in the tree to a node.

        The distance between two nodes is the number of edges that have to
            be traversed to go from one to the other.

        Args:
            center_node_id (str): The identifier of the node to which
                the distance should be determined.

        Returns:
            Dict[str, int]: The keys are node identifiers and the values
                are the corresponding distance.
        """
        neighbours = self._nodes[center_node_id].neighbouring_nodes()
        distance_dict = {center_node_id: 0}
        for node_id in neighbours:
            neighbour_distances = self._distance_to_node_rec(node_id, center_node_id)
            neighbour_distances = {node_id: distance + 1
                                   for node_id, distance
                                   in neighbour_distances.items()}
            distance_dict.update(neighbour_distances)
        return distance_dict

    def _distance_to_node_rec(self, center_node_id: str, last_node_id: str) -> Dict[str: int]:
        """
        Recursively runs through the tree to determine the distance of all nodes.
            Determines the distance of all nodes to `center_node_id` that are not in the
            subtree connecte to this node via the node with identifier `last_node_id`
        """
        neighbours = self._nodes[center_node_id].neighbouring_nodes()
        neighbours.remove(last_node_id)
        distance_dict = {center_node_id: 0}
        for node_id in neighbours:
            neighbour_distances = self._distance_to_node_rec(node_id, center_node_id)
            neighbour_distances = {node_id: distance + 1
                                   for node_id, distance
                                   in neighbour_distances.items()}
            distance_dict.update(neighbour_distances)
        return distance_dict

    def find_subtree_of_node(self, node_id: str) -> Dict[str, Node]:
        """
        Obtains the subtree from a given node towards the leaves of this tree.
        This is done recursively.

        Args:
            node_id (str): The identifier of the node from which the subtree
                should start.

        Raises:
            ValueError: If node_id is not in the tree.

        Returns:
            Dict[str, Node]: Contains the nodes of the subtree, keyed by the
                identifier. Note that this is not a Tree class object, because the root
                still has a parent.
        """

        if node_id not in self._nodes:
            err_str = f"Node with id {node_id} is not in this tree!"
            raise ValueError(err_str)

        subtree_root = self._nodes[node_id]
        subtree = {node_id: subtree_root}

        if subtree_root.is_leaf():
            # Breaking of recursion
            return subtree

        for child_id in subtree_root.children:
            # Recursion
            subtree.update(self.find_subtree_of_node(child_id))

        return subtree

    def find_subtree_size_of_node(self, node_id: str, size=0) -> int:
        """
        Obtains the subtree size from a given node

        Args:
            node_id (str): The identifier of the node from which the subtree
                should start.

        Raises:
            ValueError: If node_id is not in the tree.

        Returns:
            int: Size of subtree at node
        """

        if node_id not in self._nodes:
            err_str = f"Node with id {node_id} is not in this tree!"
            raise ValueError(err_str)
        current_node = self.nodes[node_id]

        if current_node.is_leaf():
            return 1
        size += 1
        for children_id in current_node.children:
            size += self.find_subtree_size_of_node(children_id)

        return size

    def is_child_of(self, node_id1: str, node_id2: str) -> bool:
        """
        Returns whether the node with `node_id1` is a child of the node with `node_id2`.
        """
        return self._nodes[node_id1].is_child_of(node_id2)

    def is_parent_of(self, node_id1: str, node_id2: str) -> bool:
        """
        Returns whether the node with `node_id1` is the parent of the node with `node_id2`.
        """
        return self._nodes[node_id1].is_parent_of(node_id2)

    def determine_parentage(self, node_id1: str, node_id2: str) -> Tuple[str, str]:
        """
        Orders two node identifiers by their parentage.

        Args:
            node_id1 (str): Identifier of a node in self.
            node_id2 (str): A different identifier of a node in self.

        Raises:
            ValueError: If the two nodes aren't neighbours

        Returns:
            Tuple[str, str]: The identifiers in the format (parent_id, child_id)
        """
        node1 = self._nodes[node_id1]
        node2 = self._nodes[node_id2]
        if node2.is_child_of(node_id1):
            parent_id = node_id1
            child_id = node_id2
        elif node1.is_child_of(node_id2):
            parent_id = node_id2
            child_id = node_id1
        else:
            errstr = f"Nodes {node_id1} and {node_id2} are no neighbours!"
            raise ValueError(errstr)
        return (parent_id, child_id)

    def replace_node_in_neighbours(self, new_node_id: str, old_node_id: str):
        """
        Replaces an old node with a new node for all the neighbours of
        the new node.

        Args:
            new_node_id (str): Identifier of the node to be added
            old_node_id (atr): Identifier of the node to be replaced
        """
        if new_node_id != old_node_id:
            old_node = self._nodes[old_node_id]
            for child_id in old_node.children:
                if child_id != new_node_id: # Otherwise the new node might neighbour itself
                    self._nodes[child_id].parent = new_node_id
            if old_node.is_root():
                self._root_id = new_node_id
            else:
                if old_node.parent != new_node_id:
                    self._nodes[old_node.parent].replace_child(old_node_id, new_node_id)
        self._nodes.pop(old_node_id)

    def replace_node_in_some_neighbours(self, new_node_id: str, old_node_id: str,
                                         neighbour_ids: List[str]):
        """
        Replaces an old node with a new one for some of the neighbours of
         the old node.

        Args:
            new_node_id (str): The identifier of the new node
            old_node_id (str): The identifier of the node to be replaced
            neighbour_ids (List[str]): A list of node identifiers that are neighbours
             of the old node and for which the connection to the old node is to be
             replaced by a connection to the new node. 
        """
        old_node = self._nodes[old_node_id]
        for identifier in neighbour_ids:
            neighbour_node = self._nodes[identifier]
            if not old_node.is_root() and old_node.parent == identifier:
                neighbour_node.replace_child(old_node_id, new_node_id)
            elif neighbour_node.is_child_of(old_node_id):
                neighbour_node.parent = new_node_id

    def find_path_to_root(self, node_id: str) -> List[str]:
        """
        Get a list of all parent nodes starting from node_id until root is reached.
        """
        path = [node_id] # Starting point
        while not self.nodes[node_id].is_root():
            path.append(self.nodes[node_id].parent)
            node_id = self.nodes[node_id].parent
        return path

    def path_from_to(self, start_id: str, end_id: str) -> List[str]:
        """
        Finds a path between two nodes.

        Args:
            start_id (str): Identifier of start node.
            end_id (str): Identifier of end node.

        Returns:
            List[str]: Identifiers of nodes that lie along the path. The first
             index is start_id and the last index is end_id.
        """
        sub_path_start_center = self.find_path_to_root(start_id)
        sub_path_end_center = self.find_path_to_root(end_id)
        combined = sub_path_start_center + sub_path_end_center
        num_of_duplicates = len([j for j in combined if combined.count(j) != 1])//2
        if -num_of_duplicates+1 != 0:
            sub_path_start_center_no_duplicates = sub_path_start_center[:-num_of_duplicates+1]
        else:
            sub_path_start_center_no_duplicates = sub_path_start_center[::]
        sub_path_end_center_no_duplicates = sub_path_end_center[:-num_of_duplicates]
        sub_path_end_center_no_duplicates.reverse()
        sub_path = sub_path_start_center_no_duplicates + sub_path_end_center_no_duplicates
        return sub_path
