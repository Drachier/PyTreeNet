"""
Provides the GraphNode class, which is the fundamental building block of trees.

A GraphNode is a node in a tree structure and contains the information about
neighbouring nodes, i.e. parent and children nodes.

.. code-block:: python

    # Create a graph node
    node = GraphNode("node"

    # Add a parent
    node.add_parent("parent")

    # Add children
    node.add_child("child1")
    node.add_children(["child2", "child3"])

    # We can do some checks
    node.is_root()  # False
    node.is_leaf()  # False
    node.is_child_of("parent")  # True
    node.is_parent_of("child1")  # True
    
    # An check the numbers for conneciivity
    node.nparents()  # 1
    node.nchildren()  # 3
    node.nneighbours()  # 4
    node.neighbour_index("child2") # 2
    node.child_index("child2")  # 1
    node.neighbouring_nodes()  # ["parent", "child1", "child2", "child3"]

Note that GraphNodes are usually not used directly, but rather via the Node
child class in tree tensor networks.
"""
from __future__ import annotations
from typing import List, Union
import uuid
from copy import copy

from ..util.ttn_exceptions import NoConnectionException

class GraphNode:
    """
    A graph node is the fundamental building block of a tree.

    It contains the connectivity information of the graph node in the tree
    structure, i.e. parent and children nodes.

    Attributes:
        identifier (str): A unique identifier assigned to this node.
        parent (Union[str,None]): The identifier of the parent node. There can
            only a single parent node. If there is no parent, this attribute is
            None.
        children (List[str]): A list of identifiers of the children nodes. We
            consider the order of this list to be the order of the children.
    """

    def __init__(self, identifier=""):
        """
        Creates a GraphNode.
        
        If no identifier is given, a random unique identifier is assigned.
        Initially no parent or children nodes are assigned.
        References to parent or children nodes are in the form of a node_id (str)

        Args:
            identifier (str, optional): A unique identifier assigned
                to this node. Defaults to "".
        """
        # Setting identifier
        if identifier is None or identifier == "":
            self._identifier = str(uuid.uuid1())
        else:
            self._identifier = str(identifier)

        # Information about connectivity
        self.parent: Union[str,None] = None
        self.children: List[str] = []

    def copy_with_new_id(self, new_id: str) -> GraphNode:
        """
        Creates a copy of this GraphNode with a new identifier.

        Args:
            new_id (str): The new identifier.

        Returns:
            GraphNode: A copy of this GraphNode with a new identifier.
        """
        new_node = GraphNode(new_id)
        new_node.parent = self.parent
        new_node.children = copy(self.children)
        return new_node

    @property
    def identifier(self) -> str:
        """
        A string that is unique to this node.
        """
        return self._identifier

    def __eq__(self, other: GraphNode) -> bool:
        """
        Checks if two GraphNodes are the same.

        Two GraphNodes are the same, if they have the same identifier, children
        in the right order and the same parent.
        """
        if not self.identifier == other.identifier:
            return False
        if not self.children == other.children:
            return False
        # Needed to avoid string to None comparison
        if self.is_root() and other.is_root():
            return True
        if self.is_root() or other.is_root():
            return False
        return self.parent == other.parent

    def add_parent(self, parent_id: str):
        """
        Add a parent to this node.

        Args:
            parent_id (str): The identifier of the parent node. Will be added
                as the parent of this node.
        
        Raises:
            AssertionError: If the node already has a parent. Instead remove
                the parent first and then add the new parent.
        """
        if self.parent is not None:
            errstr = f"Node {self.identifier} already has a parent!"
            raise AssertionError(errstr)
        self.parent = parent_id

    def remove_parent(self):
        """
        Removes parent and replaces it by None
        """
        self.parent = None

    def add_child(self, child_id: str):
        """
        Add a new child to this node.

        Args:
            child_id (str): The identifier of the child node. Will be added
                as a child to this node.
        """
        self.children.append(child_id)

    def add_children(self, children_ids: List[str]):
        """
        Adds mutliple children to this node.

        Args:
            children_ids (List[str]): A list of identifiers of the children
                nodes. All will be added as children to this node.
        """
        self.children.extend(children_ids)

    def _check_child_existence(self, child_id: str):
        """
        Checks, if a given identifier is that of a child of this node.

        Args:
            child_id (str): The identifier to check.

        Raises:
            ValueError: If the child_id is not a child of this node.
        """
        if child_id not in self.children:
            errstr = f"{child_id} is not a child of this node ({self.identifier})!"
            raise ValueError(errstr)

    def remove_child(self, child_id: str):
        """
        Removes a child from this node's children.

        Args:
            child_id (str): The identifier of the child to be removed.
        """
        self._check_child_existence(child_id)
        self.children.remove(child_id)

    def child_index(self, child_id: str) -> int:
        """
        Returns the index of a child of the node.

        The order is (usually) defined by the order in which the children
        were added.

        Args:
            child_id (str): The identifier of the child to look for.
        """
        self._check_child_existence(child_id)
        return self.children.index(child_id)

    def neighbour_index(self, node_id: str) -> int:
        """
        Returns the index of the neighbour of this node.

        This includes the parent with index 0 and the children.

        Args:
            node_id (str): The identifier of the node to look for.

        Returns:
            int: The index of the node.
        """
        if node_id == self.parent:
            return 0
        if node_id in self.children:
            return self.children.index(node_id) + self.nparents()
        errstr = f"{node_id} is not a neighbour of {self.identifier}!"
        raise NoConnectionException(errstr)

    def replace_child(self, child_id: str, new_child_id: str):
        """
        Replaces one child with another.

        This is done in place.

        Args:
            child_id (str): The identifier of the child to be replaced.
            new_child_id (str): The identifier of the new child.
        """
        self._check_child_existence(child_id)
        if child_id == new_child_id:
            return
        self.children[self.child_index(child_id)] = new_child_id

    def replace_neighbour(self, old_neighbour_id: str, new_neighbour_id: str):
        """
        Replaces a neighbour with a new one.

        This is done in place.

        Args:
            old_neighbour_id (str): The node identifier to be replaced.
            new_neighbour_id (str): The new node identifier to be inserted.
        """
        if not self.is_root() and self.parent == old_neighbour_id:
            self.parent = new_neighbour_id
        elif old_neighbour_id in self.children:
            self.replace_child(old_neighbour_id,new_neighbour_id)
        else:
            errstr = f"{old_neighbour_id} is not a neighbour of {self.identifier}!"
            raise NoConnectionException(errstr)

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

        This includes the children and the parent.
        """
        return self.nchildren() + (not self.is_root())

    def has_x_children(self, x: int) -> bool:
        """
        Returns whether this node has exactly x-many children.
        """
        return self.nchildren() == x

    def neighbouring_nodes(self) -> List[str]:
        """
        Provides the identifiers of all neighbours.

        This means of all children and the parent, if it exists.

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
