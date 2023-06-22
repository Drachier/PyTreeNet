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

    def _add_node(self, new_node: Node):
        """
        Adds a node to the dictionary with checks.

        Only for internal use. For external additions use the functions below.
        """
        new_node_id = new_node.identifier
        if new_node_id in self._nodes:
            err_str = f"Tree already contains a node with identifier {new_node_id}!"
            raise ValueError(err_str)
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
        if parent_id not in self._nodes:
            err_str = f"Node with identifier {parent_id} is not in this tree!"
            raise ValueError(err_str)

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

    def nearest_neighbour(self) -> List[Tuple[str, str]]:
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
        neighbours = self[center_node_id].neighbouring_nodes()
        distance_dict = {center_node_id: 0}
        for node_id in neighbours:
            neighbour_distances = self.distance_to_node(node_id)
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

    def combine_nodes(self, node_id1: str, node_id2: str, new_identifier: str=""):
        """
        Combines the two nodes with the given identifiers.

        The children in the new node have the form
            [parent_children, child_children]

        Args:
            node_id1 (str): Identifier of the first node.
            node_id2 (str): Identifier of the second node.
            new_identifier (str): A potential new identifier. Otherwise defaults to
                `parent_id + 'contr' + child_id`
        """
        parent_id, child_id = self.determine_parentage(node_id1, node_id2)
        if new_identifier == "":
            new_identifier = parent_id + "contr" + child_id

        # Find new neighbours
        ## Parent
        parent_node = self._nodes[parent_id]
        parent_parent_id = parent_node.parent
        ## Children
        parent_node.remove_child(child_id)
        total_children = parent_node.children
        child = self._nodes[child_id]
        total_children.extend(child.children)

        # Remove old nodes
        self._nodes.pop(parent_id)
        self._nodes.pop(child_id)

        # Build new node
        new_node = Node(identifier=new_identifier)
        new_node.add_children(total_children)
        new_node.add_parent(parent_parent_id)
        self._nodes[new_node.identifier] = new_node

        # Change connectivity
        for node_id in new_node.children:
            node = self._nodes[node_id]
            node.remove_parent()
            node.add_parent(new_node.identifier)
        if parent_parent_id is not None:
            node = self._nodes[parent_parent_id]
            node.remove_child(parent_id)
            node.add_child(new_node.identifier)

    # def rewire_only_child(self, parent_id: str, child_id: str, new_identifier: str):
    #     """
    #     For the node with identifier child_id the parent_leg is rewired from parent
    #     to a node with identifier new_identifier.

    #     Parameters
    #     ----------
    #     parent_id : str
    #         Identifier of the parent node for which one child is rewired to a new parent.
    #     child_id : str
    #         Identifier of the child which is to be rewired.
    #     new_identifier : str
    #         Identifier of the node to be rewired to.

    #     Returns
    #     -------
    #     None.

    #     """
    #     parent = self.nodes[parent_id]
    #     child = self.nodes[child_id]
    #     assert child_id in parent.children_legs, f"The node with identifier {child_id} is not a child of the node with identifier {parent_id}."
    #     assert child.parent_leg[
    #         0] == parent_id, f"The node with identifier {parent_id} is not the parent of the node with identifier {child_id}."
    #     child.parent_leg[0] = new_identifier

    # def rewire_only_parent(self, child_id: str, new_identifier: str):
    #     """
    #     For the parent of the node child the leg connected to child is rewired to the
    #     tensor node with identifier new_identifier.

    #     Parameters
    #     ----------
    #     child_id : str
    #         Identifier of the node whose parent is to have one leg rewired.
    #     new_identifier : str
    #         Identifier of the tensor the parent is rewired to.

    #     Returns
    #     -------
    #     None.

    #     """
    #     child = self.nodes[child_id]
    #     if child.is_root():
    #         warn(
    #             f"The node with identifier {child_id} is a tree's root, so its parent cannot be rewired.")
    #     else:
    #         parent_id = child.parent_leg[0]
    #         parent = self.nodes[parent_id]
    #         leg_to_child_tensor = {
    #             new_identifier: parent.children_legs[child_id]}
    #         del parent.children_legs[child_id]
    #         parent.children_legs.update(leg_to_child_tensor)
