from __future__ import annotations

class Tree(object):
    """
    A rooted tree that contains nodes, which in turn contain
    the information on how they are connected to each other.

    Attributes
    ----------
    _nodes (dict[str, Node]): A dictionary mapping node identifiers to Node objects.
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
