import numpy as np
import copy

class TreeTensorNetwork(object):
    """
    A tree tensor network (TTN) a tree, where each node contains a tensor,
    that is part of the network. Here a tree tensor network is a dictionary
    _nodes of tensor nodes with their identifiers as keys.
    
    General structure and parts of the codes are from treelib.tree
    """

    def __init__(self, original_tree = None, deep = False):
        """
        Initiates a new TreeTensorNetwork or a deep or shallow copy of a
        different one.
        """
        
        self._nodes = dict()
        
        self._root = None
        if original_tree is not None:
            self._root = original_tree.root
            
            if deep:
                for node_id in original_tree.nodes:
                    self._nodes[node_id] = copy.deepcopy(original_tree.nodes[node_id])
                else:
                    self._nodes = original_tree.nodes
        
        @property
        def nodes(self):
            """
            A dictionary containing the tensor trees notes via their identifiers.
            """
            return self._nodes
        
        @property
        def root(self):
            """
            The root's identifier.'
            """
            return self._root

        def __contains__(self, identifier):
            """
            Determines if a node with identifier is in the TTN.
            """
            return identifier in self._nodes
        
        def __getitem__(self, key):
            """
            Return _nodes[key]
            """
            return self._nodes[key]
        
        def __len__(self):
            return len(self._nodes)
            
        def __setitem__(self, node):
            "TODO: Once the update methods are established this can work."
        
        def add_node(self, node, parent=None):
            """
            Adds a tensor in form of a tensor node to the TNN as a child of
            parent (the identifier of another node).
            If parent=None the new node becomes the root, if there isn't one
            yet.
            """
             if parent == None:
                 
        
        
            