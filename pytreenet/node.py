
from uuid import uuid1

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
    
