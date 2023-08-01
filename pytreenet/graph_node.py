from __future__ import annotations
import uuid

import numpy as np

from typing import List

from .util import crandn, copy_object


class GraphNode:
    """
    A graph node is the fundamental building block of a tree.
    It contains the connectivity information of the graph node in the 
    tree tensor network structure.
    """

    def __init__(self, tag="", identifier=""):
        """
        Creates a GraphNode. If no identifier is given, a random
        unique identifier is assigned.
        Initially no parent or children nodes are assigned.
        References to parent or children nodes are in the form of a node_id (str)

        Args:
            tag (str, optional): A non-unique name of the node.
                Defaults to "".
            identifier (str, optional): A unique identifier assigned
                to this node. Defaults to "".
        """
        # Setting identifier
        if identifier is None or identifier == "":
            self._identifier = str(uuid.uuid1())
        else:
            self._identifier = str(identifier)
        # Setting tag
        if tag is None or tag == "":
            self._tag = self.identifier
        else:
            self._tag = str(tag)

        # Information about connectivity
        self.parent = None
        self.children = []

    @property
    def identifier(self):
        """
        A string that is unique to this node.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, new_identifier):
        if new_identifier is None or new_identifier == "":
            self._identifier = str(uuid.uuid1())
        else:
            self._identifier = str(new_identifier)

    @property
    def tag(self):
        """
        A human readable tag for this node.
        """
        return self._tag

    @tag.setter
    def tag(self, new_tag):
        if new_tag is None or new_tag == "":
            self._tag = self.identifier
        else:
            self._tag = new_tag

    def __eq__(self, other: GraphNode):
        """
        Two nodes are the same, if they have the same identifiers and neighbours.
        """
        same_id = self.identifier == other.identifier
        same_parent = self.parent == other.parent
        same_children = self.children == other.children
        return same_id and same_parent and same_children

    def add_parent(self, parent_id: str):
        """
        Adds `parent_id` as the new parent.

        If a parent already exists, remove it and then add the new one.
        """
        if self.parent is not None:
            errstr = f"Node {self.identifier} has a parent already"
            raise ValueError(errstr)
        self.parent = parent_id

    def remove_parent(self):
        """
        Removes parent and replaces it by None
        """
        self.parent = None

    def add_child(self, child_id: str):
        """
        Adds `child_id` as a new child.
        """
        self.children.append(child_id)

    def add_children(self, children_ids: List[str]):
        """
        Adds all children with identifiers in `children_ids` as children to this node.
        """
        self.children.extend(children_ids)

    def remove_child(self, child_id: str):
        """
        Removes a children identifier from this node's children.
        """
        try:
            self.children.remove(child_id)
        except ValueError as exc:
            errstr = f"{child_id} is not a child of this node!"
            raise ValueError(errstr) from exc

    def child_index(self, child_id: str) -> int:
        """
        Returns the index of identifier child_id in this Node's children list.
        """
        try:
            return self.children.index(child_id)
        except ValueError as exc:
            errstr = f"{child_id} is not a child of this node!"
            raise ValueError(errstr) from exc

    def replace_child(self, child_id: str, new_child_id: str):
        """
        Replaces one child with another.
        """
        if child_id == new_child_id:
            return
        self.children[self.child_index(child_id)] = new_child_id

    def is_root(self) -> bool:
        """
        Returns whether this node is a root node, i.e. doesn't have a parent.
        """
        return self.parent is None

    def is_leaf(self) -> bool:
        """
        Returns whether this node is a leaf, i.e. doesn't have children
        """
        return self.has_x_children(0)

    def is_child_of(self, other_node_id: str) -> bool:
        """
        Determines whether this node is a child of the node with identifier 'other_node_id'.
        """
        if self.is_root():
            return False
        return self.parent == other_node_id

    def is_parent_of(self, other_node_id: str) -> bool:
        """
        Determines whether this node is a parent of the node with identifier 'other_node_id'.
        """
        return other_node_id in self.children

    def nparents(self) -> int:
        """
        Returns the number of parents of this node.
        """
        return int(not self.is_root())

    def nchildren(self) -> int:
        """
        The number of children of this node
        """
        return len(self.children)

    def nneighbours(self) -> int:
        """
        Returns the number of neighbours of this node.
        """
        return self.nchildren() + (not self.is_root())

    def has_x_children(self, x: int) -> bool:
        """
        Returns whether this node has exactly x-many children.
        """
        return len(self.children) == x

    def neighbouring_nodes(self) -> List[str]:
        """
        Provides the identifiers of all neighbours, i.e. the parent and all
            children.

        Returns:
            List[str]: Contains the neighbour identifiers, if this node is not
                a root, the parent's identifier is the first identifier.
        """
        if self.is_root():
            neighbour_ids = []
        else:
            neighbour_ids = [self.parent]
        neighbour_ids.extend(self.children)
        return neighbour_ids


def assert_legs_matching(node1: GraphNode, leg1_index: int, node2: GraphNode, leg2_index: int):
    """
    Asserts if the dimensions of leg1 of node1 and leg2 of node2 match.
    """
    leg1_dimension = node1.shape[leg1_index]
    leg2_dimension = node2.shape[leg2_index]
    assert leg1_dimension == leg2_dimension
