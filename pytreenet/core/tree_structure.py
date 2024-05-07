"""
This module deals with the graph connectivity of a tree.

Essential to this is the tree structure class, that acts as a collection of all
nodes of a tree. The methods provide various ways to gather information and
change the connectivity of the tree.


"""

from __future__ import annotations
from typing import List, Tuple, Dict, Union

from .graph_node import GraphNode
from ..util.ttn_exceptions import NoConnectionException


class TreeStructure():
    """
    The class holding the structure and connectivity of a tree.

    Acts as a collection of nodes. These can be added and removed in relation
    to another. Furthermore information about the graph structure and
    connectivity can be extracted.

    Attributes:
        nodes (Dict[str,GraphNode]): Hold the nodes belonging to the tree.
            It should be modified using the appropriate methods.
        root_id [str,None]: The identifier of the root node. This defines a
            direction on the tree. It is only None, if the tree contains no
            node.
    """

    def __init__(self):
        """
        Initiates a new and empty TreeStructure.
        """
        self._nodes: Dict[str,GraphNode] = {}
        self._root_id: Union[str,None] = None

    @property
    def nodes(self) -> Dict[str, GraphNode]:
        """
        Returns the nodes of this tree structure.

        Can only be modified using the appropriate methods of the tree
        structure.
        """
        return self._nodes

    @property
    def root_id(self) -> Union[None, str]:
        """
        Returns the root's identifier.
        """
        return self._root_id

    def __contains__(self,
                     identifier: Union[str,GraphNode]) -> bool:
        """
        Determines if a node is in the TreeStructure.

        Args:
            identifier (Union[str,GraphNode]): Either the identifier or the
                the node itself. However, even if the graph node is provided
                only the identifier is checked.
        """
        if isinstance(identifier,GraphNode):
            identifier = identifier.identifier
        return identifier in self._nodes

    def __getitem__(self, key: str) -> GraphNode:
        """
        Return Node object associated with `key`
        """
        return self._nodes[key]

    def __len__(self) -> int:
        """
        Returns the length of a tree structure.

        This is defined as the number of nodes in the tree,
        """
        return len(self._nodes)

    def ensure_uniqueness(self, node_id: str):
        """
        Ensures that the given identifier is not already in use.

        Args:
            node_id (str): Identifer to check.

        Raises:
            ValueError: Raised if the a node in this tree already has this
                identifier.
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
            ValueError: Raised if there is no node with the identifier in this
                tree.
        """
        if node_id not in self._nodes:
            err_str = f"Node with identifier {node_id} is not in this tree!"
            raise ValueError(err_str)

    def _add_node(self, new_node: GraphNode):
        """
        Adds a node to the dictionary with checks.

        Only for internal use. For external additions use the methods below.

        Args:
            new_node (GraphNode): The node to be added.
        """
        new_node_id = new_node.identifier
        self.ensure_uniqueness(new_node_id)
        self.nodes[new_node_id] = new_node

    def add_root(self, node: GraphNode):
        """
        Adds a root node to this tree.

        Args:
            node (GraphNode): The node to be added.
        """
        assert self.root_id is None, "A tree may only have one root."
        self._root_id = node.identifier
        self._nodes[node.identifier] = node

    def add_child_to_parent(self, child: GraphNode, parent_id: str):
        """
        Adds a Node as a child to the specified parent_node.

        Args:
            child (GraphNode): The node to be added
            parent_id (str): The identifier of the node which is to be the new
                parent.
        """
        self.ensure_existence(parent_id)
        self._add_node(child)
        child.add_parent(parent_id)

        child_id = child.identifier
        parent = self._nodes[parent_id]
        parent.add_child(child_id)

    def add_parent_to_root(self, new_root: GraphNode):
        """
        Adds a parent to the root of this tree, making it the new root.

        Args:
            node (GraphNode): The node to be added.
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

        Returns:
            nn_list (List[Tuple[str,str]]) : A list containing the identifiers of all
                nearest neighbour pairs. The first element of the tuple is the parent
                and the second the child. The order in the list is the same as the order
                of the nodes saved in the tree.
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
        Finds the distance of every node in the tree to a given node.

        The distance between two nodes is the number of edges that have to be
        traversed to go from one to the other.

        Args:
            center_node_id (str): The identifier of the node to which the
                distance should be determined.

        Returns:
            Dict[str, int]: The keys are node identifiers and the values are
                the corresponding distance. The root will be the first entry
                followed by the first child, followed by the first grandchild.
                Once a leaf is reached, the closest unvisited node will be
                next.
        """
        self.ensure_existence(center_node_id)
        neighbours = self._nodes[center_node_id].neighbouring_nodes()
        distance_dict = {center_node_id: 0}
        for node_id in neighbours:
            neighbour_distances = self._distance_to_node_rec(node_id, center_node_id)
            neighbour_distances = {node_id: distance + 1
                                   for node_id, distance
                                   in neighbour_distances.items()}
            distance_dict.update(neighbour_distances)
        return distance_dict

    def _distance_to_node_rec(self,
                              center_node_id: str,
                              last_node_id: str) -> Dict[str: int]:
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

    def find_subtree_of_node(self, node_id: str) -> Dict[str, GraphNode]:
        """
        Obtains the subtree from a given node towards the leaves of this tree.
        
        This is done recursively.

        Args:
            node_id (str): The identifier of the node from which the subtree
                should start.

        Returns:
            Dict[str, Node]: Contains the nodes of the subtree, keyed by the
                identifier. Note that this is not a Tree class object, because
                the root still has a parent.
        """
        self.ensure_existence(node_id)
        subtree_root = self._nodes[node_id]
        subtree = {node_id: subtree_root}
        if subtree_root.is_leaf():
            # Breaking of recursion
            return subtree
        for child_id in subtree_root.children:
            # Recursion
            subtree.update(self.find_subtree_of_node(child_id))
        return subtree

    def leaves_under_node(self, node_id: str) -> Dict[str, GraphNode]:
        """
        Recursively finds all leaves below a given node.

        Args:
            node_id (str): Node from which to start

        Returns:
            Dict[str, GraphNode]: Contains all leaves that are in this tree
                below the given node.
        """
        self.ensure_existence(node_id)
        node = self.nodes[node_id]
        if node.is_leaf():
            return {node_id: node}
        leaves = {}
        for child_id in node.children:
            leaves.update(self.leaves_under_node(child_id))
        return leaves

    def find_subtree_size_of_node(self, node_id: str, size=0) -> int:
        """
        Obtains the subtree size from a given node

        Args:
            node_id (str): The identifier of the node from which the subtree
                should start.

        Returns:
            int: Size of subtree at node
        """
        self.ensure_existence(node_id)
        current_node = self.nodes[node_id]

        if current_node.is_leaf():
            return 1
        size += 1
        for children_id in current_node.children:
            size += self.find_subtree_size_of_node(children_id)

        return size

    def is_child_of(self, pot_child: str, pot_parent: str) -> bool:
        """
        Tests if one node is a child of the other.

        Args:
            pot_child (str): Identifier of the potential child.
            pot_parent (str): Identifier of the potential parent.

        Returns:
            bool: True if the first node is a child of the second.
        """
        if self._nodes[pot_child].is_child_of(pot_parent):
            assert self._nodes[pot_parent].is_parent_of(pot_child)
            return True
        return False

    def is_parent_of(self, pot_parent: str, pot_child: str) -> bool:
        """
        Tests if one node is a parent of the other.

        Args:
            pot_parent (str): Identifier of the potential parent.
            pot_child (str): Identifier of the potential child.
        
        Returns:
            bool: True if the first node is a parent of the second.
        """
        return self.is_child_of(pot_child, pot_parent)

    def determine_parentage(self,
                            node_id1: str,
                            node_id2: str) -> Tuple[str, str]:
        """
        Orders two node identifiers by their parentage.

        Args:
            node_id1 (str): Identifier of one node.
            node_id2 (str): A different identifier of a second node.

        Returns:
            Tuple[str, str]: The identifiers are in the forma
                `(parent_id, child_id)`
        """
        node1 = self._nodes[node_id1]
        node2 = self._nodes[node_id2]
        if node2.is_child_of(node_id1):
            return (node_id1, node_id2)
        if node1.is_child_of(node_id2):
            return (node_id2, node_id1)
        errstr = f"Nodes {node_id1} and {node_id2} are no neighbours!"
        raise NoConnectionException(errstr)

    def replace_node_in_neighbours(self, new_node_id: str, old_node_id: str):
        """
        Replaces a node in all neighbours of the old node.

        Args:
            new_node_id (str): Identifier of the node to be added
            old_node_id (atr): Identifier of the node to be replaced
        """
        if new_node_id != old_node_id:
            old_node = self._nodes[old_node_id]
            for child_id in old_node.children:
                if child_id != new_node_id:
                    # Otherwise the new node might neighbour itself
                    self._nodes[child_id].parent = new_node_id
            if old_node.is_root():
                self._root_id = new_node_id
            else:
                if old_node.parent != new_node_id:
                    # Otherwise the new node might neighbour itself
                    self._nodes[old_node.parent].replace_child(old_node_id, new_node_id)
            self._nodes.pop(old_node_id)

    def replace_node_in_some_neighbours(self, new_node_id: str, old_node_id: str,
                                         neighbour_ids: List[str]):
        """
        Replaces a node for only some of its neighbours.

        Args:
            new_node_id (str): The identifier of the new node
            old_node_id (str): The identifier of the node to be replaced
            neighbour_ids (List[str]): A list of node identifiers that are neighbours
             of the old node and for which the connection to the old node is to be
             replaced by a connection to the new node. 
        """
        for neighbour_id in neighbour_ids:
            neighbour = self.nodes[neighbour_id]
            neighbour.replace_neighbour(old_node_id, new_node_id)

    def find_path_to_root(self, node_id: str) -> List[str]:
        """
        Finds the path of a node to the root.

        Args:
            node_id (str): The node from which to start the path.
        
        Returns:
            List[str]: A list of all nodes on the path. The first entry is the
                starting node and the last entry is the root node.
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
             indentifier is `start_id` and the last indentifier is `end_id`.
        """
        if start_id == end_id:
            return [start_id]
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

    def linearise(self) -> List[str]:
        """
        Linearises the tree.

        Returns:
            List[str]: The identifiers of the nodes in the order they are
                visited. The children of a node will always appear before the
                node itself.
        """
        linearised = []
        if not self._root_id is None:
            self._linearised_rec(self._root_id, linearised)
        return linearised

    def _linearised_rec(self, node_id: str, linearised: List[str]):
        """
        The recursive part of the linearise function.

        Args:
            node_id (str): The identifier of the node to be linearised.
            linearised (List[str]): The list to which the identifiers are added.
        """
        for child_id in self._nodes[node_id].children:
            self._linearised_rec(child_id, linearised)
        linearised.append(node_id)
