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
from typing import List, Union, Callable
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

    def set_identifier(self, new_id: str):
        """
        Sets a new identifier for this node.

        Should only happen very explicitly, as this can lead to inconsistencies

        Args:
            new_id (str): The new identifier.
        """
        self._identifier = new_id

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

    def is_child_of(self, other_node: str| GraphNode) -> bool:
        """
        Determines whether this node is a child of the a given node.

        Args:
            other_node_id (str|GraphNode): The identifier of the node to check
                against or the GraphNode itself.
        
        Returns:
            bool: True if this node is a child of the given node, False otherwise.
        """
        if self.is_root():
            return False
        if isinstance(other_node, GraphNode):
            other_node = other_node.identifier
        return self.parent == other_node

    def is_parent_of(self, other_node: str | GraphNode | list[str | GraphNode
                                                              ]) -> bool:
        """
        Determines whether this node is a parent of a given node.

        Args:
            other_node (str | GraphNode | List[str | GraphNode]): The
                identifier of the node to check against or the GraphNode
                itself.
            
        Returns:
            bool: True if this node is a parent of the given node(s), False
                otherwise.
        """
        if isinstance(other_node, list):
            return all(self.is_parent_of(node)
                       for node in other_node)
        if isinstance(other_node, GraphNode):
            other_node = other_node.identifier
        return other_node in self.children

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

    def neighbour_id(self,
                     index: int
                     ) -> str:
        """
        Returns the identifier of the neighbour at the given index.

        Args:
            index (int): The index of the neighbour.

        Returns:
            str: The identifier of the neighbour at the given index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= self.nneighbours():
            raise IndexError("Index out of bounds!")
        if not self.is_root():
            if index == 0:
                return self.parent
            return self.children[index - 1]
        return self.children[index]

## Functions to using two or more GraphNodes together
def determine_parentage(node1: GraphNode,
                        node2: GraphNode
                        ) -> tuple[GraphNode, GraphNode]:
    """
    Sorts two nodes by parentage.

    Args:
        node1 (GraphNode): The first node.
        node2 (GraphNode): The second node.

    Returns:
        tuple[GraphNode, GraphNode]: A tuple containing the parent node first
            and the child node second.
    
    Raises:
        ValueError: If the nodes are not connected.
    """
    if node2.is_child_of(node1):
        return (node1, node2)
    if node1.is_child_of(node2):
        return (node2, node1)
    errstr = f"Nodes {node1.identifier} and {node2.identifier} are no neighbours!"
    raise NoConnectionException(errstr)

def find_children_permutation(old_node: GraphNode,
                              new_node: GraphNode,
                              modify_function: Union[Callable,None] = None
                              ) -> List[int]:
    """
    Finds a permutation of children of one node to the other.

    The permutation found is the permutation required to transform the children
    of the old node to the children of the new node.

    .. code-block:: python

        old_node = GraphNode("old")
        old_node.add_children(["child1", "child2", "child3"])
        new_node = GraphNode("new")
        new_node.add_children(["child2", "child3", "child1"])
        perm = find_children_permutation(old_node, new_node)
        print(perm)  # [1, 2, 0]


    Args:
        old_node (GraphNode): The old node.
        new_node (GraphNode): The new node.
        modify_function (Callable): A function that modifies the children
            identifiers of the new node.

    Returns:
        List[int]: The permutation of the children of the old node to the
            children of the new node.
    """
    if len(old_node.children) != len(new_node.children):
        raise ValueError("The number of children must be the same!")
    if modify_function is None:
        return [old_node.children.index(child_id)
                for child_id in new_node.children]
    return [old_node.children.index(modify_function(child_id))
            for child_id in new_node.children]

def find_child_permutation_neighbour_index(old_node: GraphNode,
                                            new_node: GraphNode,
                                            modify_function: Union[Callable,None] = None
                                            ) -> List[int]:
    """
    Finds a permutation of children of one node to the other according to their
    neighbour indices.

    The permutation found is the permutation required to transform the children
    of the old node to the children of the new node.

    .. code-block:: python

        old_node = GraphNode("old")
        old_node.add_children(["child1", "child2", "child3"])
        old_node.add_parent("parent")
        new_node = GraphNode("new")
        new_node.add_children(["child2", "child3", "child1"])
        new_node.add_parent("parent")
        perm = find_children_permutation(old_node, new_node)
        print(perm)  # [2, 3, 1]


    Args:
        old_node (GraphNode): The old node.
        new_node (GraphNode): The new node.
        modify_function (Callable): A function that modifies the children
            identifiers of the new node.

    Returns:
        List[int]: The permutation of the children of the old node to the
            children of the new node, according to their neighbour index.
    """
    child_perm = find_children_permutation(old_node,
                                           new_node,
                                           modify_function)
    assert old_node.nparents() == new_node.nparents()
    nparents = old_node.nparents()
    return [num+nparents for num in child_perm]
