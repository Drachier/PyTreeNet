import numpy as np
import treelib as tl

class tensornode(tl.Node):
    """
    Subclasses the Node object by adding a tensor associated to the node and
    determining the open legs and the legs to the parent and children.
    """
    
    def __init__(self, tensor, tag=None, identifier=None, expanded=True, data=None):
        self._tensor = tensor
        super().__init__(tag=tag, identifier=identifier, expanded=expanded, data=data)
        
        self._open_legs = np.arange(tensor.ndim)
        self._parent_leg = None
        self._children_legs = dict()