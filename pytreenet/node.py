from __future__ import annotations

from uuid import uuid1
from typing import List


class Node(object):
    """
    A node is the fundamental building block of a tree.
    It contains all the information on how it is connected
    to the rest of the tree.
    """

    def __init__(self, identifier="") -> Node:
        """
        Creates a Node. If no identifier is given, a random
        unique identifier is assigned.
        Initially no parent or children nodes are assigned.
        Both attributes would contain the identifier of other nodes.

        Args:
            identifier (str, optional): A unique identifier assigned
                to this node. Defaults to "".

        Returns:
            Node: 
        """

        # Setting the identifier
        if identifier == "" or identifier is None:
            self._identifier = str(uuid1())
        else:
            self._identifier = identifier

        # Information about connectivity
        self.parent = None
        self.children = []

    @property
    def identifier(self):
        """
        Get the identifier.
        """
        return self._identifier

    def __repr__(self) -> str:
        repr_dict = {"identifier": self._identifier}
        repr_dict["parent"] = self.parent
        repr_dict["children"] = self.children
        return str(repr_dict)

    def __str__(self) -> str:
        string = "identifier: " + self._identifier + "\n"
        string += "parent: " + self.parent + "\n"
        string += "children: " + self.children + "\n"
        return string

    def is_root(self) -> bool:
        """
        Returns whether this node is a root node, i.e. doesn't have a parent.
        """
        return self.parent is None

    def has_x_children(self, x: int) -> bool:
        """
        Returns whether this node has exactly x-many children.
        """
        return len(self.children) == x

    def is_leaf(self) -> bool:
        """
        Returns whether this node is a leaf, i.e. doesn't have children
        """
        return self.has_x_children(0)

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

    def is_child_of(self, other_node_id: str) -> bool:
        """
        Determines whether this node is a child of the node with identifier 'other_node_id'.
        """
        if self.is_root():
            return False
        return self.parent is other_node_id

    def is_parent_of(self, other_node_id: str) -> bool:
        """
        Determines whether this node is a parent of the node with identifier 'other_node_id'.
        """
        return other_node_id in self.children
