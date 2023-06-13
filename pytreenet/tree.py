from __future__ import annotations


class Tree(object):
    """
    A rooted tree that contains nodes, which in turn contain
    the information on how they are connected to each other.

    Attributes
    ----------
    _nodes (dict[str, Node]): A dictionary mapping node identifiers to Node objects.
    _root_id (str): The identifier of the root node.
    """

    def __init__(self):
        """
        Initialises a Tree with no initial nodes.
        """

        self._nodes = {}
        self._root_id = None

    @property
    def nodes(self):
        """
        A dictionary containing the nodes of the tree via their identifiers.
        """
        return self._nodes

    @property
    def root_id(self):
        """
        The identifier of the root node.
        """
        return self._root_id

    def __contains__(self, identifier: str):
        """
        Determines if a node with 'identifier' is contained in the Tree.
        """
        return identifier in self.nodes

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

        child_id = child.identifier
        if child_id in self._nodes:
            err_str = f"Tree already contains a node with identifier {child_id}!"
            raise ValueError(err_str)
        self._nodes[child_id] = child
        child.add_parent(parent_id)

        parent = self._nodes[parent_id]
        parent.add_child(child_id)

    def add_parent_to_root(self, new_root: Node):
        """
        Adds a parent to the root of this tree, making it the new root.
        """
        new_id = new_root.identifier
        if new_id in self._nodes:
            err_str = f"Tree already contains a node with identifier {new_id}!"
            raise ValueError(err_str)
        self._nodes[new_id] = new_root
        new_root.add_child(self._root_id)

        current_root = self._nodes[self._root_id]
        current_root.add_parent(new_id)

        self._root_id = new_id
